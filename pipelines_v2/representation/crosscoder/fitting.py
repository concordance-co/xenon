"""Pinned reference fitting with safe, complete CPU checkpoint state."""
from dataclasses import asdict,dataclass
import random
import numpy as np
from pipelines_v2.core.provenance import content_hash
from .alignment import validate_pairs
from .reference import reference_modules,lock,verify_runtime
from .settings import Execution,Limits,numerical_scope,observed_policy,check_deadline


@dataclass(frozen=True)
class FitConfig:
    width:int=16
    steps:int=20
    batch_size:int=16
    seed:int=0
    learning_rate:float=.001
    l1_penalty:float=.001
    warmup_steps:int=1
    def __post_init__(self):
        import math
        if any(type(v)is not int or v<1 for v in (self.width,self.steps,self.batch_size,self.warmup_steps)):
            raise ValueError('Fit width, steps, batch and warmup must be positive integers')
        if type(self.seed)is not int or self.seed<0:raise ValueError('Explicit nonnegative fit seed required')
        if any(type(v)not in (int,float) or not math.isfinite(v) or v<=0 for v in (self.learning_rate,self.l1_penalty)):
            raise ValueError('Finite positive fitting rates required')


def pack(value):
    import torch
    if isinstance(value,torch.Tensor):return {'tensor':value.detach().cpu().numpy().copy(),'stride':list(value.stride())}
    if isinstance(value,np.ndarray):return {'ndarray':value.copy()}
    if isinstance(value,dict):return {'mapping':[[pack(k),pack(v)] for k,v in value.items()]}
    if isinstance(value,(list,tuple)):return {'tuple' if isinstance(value,tuple) else 'list':[pack(v) for v in value]}
    if value is None or type(value)in(str,int,float,bool):return value
    raise TypeError('Unsafe checkpoint state type: '+type(value).__name__)


def unpack(value):
    import torch
    if not isinstance(value,dict):return value
    if set(value)=={'tensor','stride'}:
        array=np.asarray(value['tensor']);tensor=torch.empty_strided(array.shape,tuple(value['stride']),dtype=torch.from_numpy(array).dtype)
        return tensor.copy_(torch.from_numpy(array.copy()))
    if set(value)=={'ndarray'}:return np.asarray(value['ndarray']).copy()
    if set(value)=={'mapping'}:return {unpack(k):unpack(v) for k,v in value['mapping']}
    if set(value)=={'tuple'}:return tuple(unpack(v) for v in value['tuple'])
    if set(value)=={'list'}:return [unpack(v) for v in value['list']]
    raise ValueError('Invalid safe checkpoint structure')


def tree_digest(value):
    if isinstance(value,np.ndarray):
        import hashlib
        return {'array_sha256':hashlib.sha256(value.tobytes(order='C')).hexdigest(),'shape':list(value.shape),'dtype':value.dtype.str}
    if isinstance(value,dict):return {k:tree_digest(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [tree_digest(v) for v in value]
    return value


def validate_checkpoint(checkpoint,expected_identity=None):
    body={k:v for k,v in checkpoint.items() if k!='checkpoint_digest'}
    if checkpoint.get('checkpoint_digest')!=content_hash(tree_digest(body)):
        raise ValueError('Crosscoder checkpoint content changed')
    if checkpoint.get('schema_version')!=1 or checkpoint.get('kind')!='crosscoder_tensors':raise ValueError('Unsupported checkpoint format')
    if expected_identity is not None and checkpoint['fit_identity']!=expected_identity:raise ValueError('Foreign alignment/config/runtime checkpoint')
    if not 0<=checkpoint['step']<=checkpoint['config']['steps']:raise ValueError('Invalid checkpoint sampler position')
    return checkpoint


def trainer_for(reference_path,config,alignment,execution=None):
    execution=execution or Execution()
    _,module=reference_modules(reference_path)
    width=len(alignment['model_sides'])
    return module.CrossCoderTrainer(activation_dim=alignment['hidden_size'],dict_size=config.width,num_layers=width,
        lr=config.learning_rate,l1_penalty=config.l1_penalty,warmup_steps=config.warmup_steps,seed=config.seed,
        device=execution.device,layer=alignment['layer'],lm_name='ordered-pinned-qwen2.5-pair',use_mse_loss=True,
        dict_class_kwargs={'norm_init_scale':.005,'code_normalization':'crosscoder'},resample_steps=None,compile=False)


def training_batch(pairs, keys, offsets, indices):
    """Gather only the sampled token rows; never concatenate the full corpus."""
    hidden = np.asarray(pairs['records'][keys[0]]['values']).shape[-1]
    batch = np.empty((len(indices), 2, hidden), dtype=np.float32)
    row_indices = np.searchsorted(offsets, indices, side='right') - 1
    for row_index in np.unique(row_indices):
        selected = np.flatnonzero(row_indices == row_index)
        batch[selected] = pairs['records'][keys[row_index]]['values'][indices[selected] - offsets[row_index]]
    return batch


def fit(pairs, reference_path, config, *, checkpoint=None, stop_after=None,
        on_checkpoint=None, verify_environment=True, execution=None, limits=None):
    import torch
    execution = execution or Execution()
    limits = limits or Limits()
    deadline = limits.deadline()
    reference_modules(reference_path)
    alignment = validate_pairs(pairs)
    hidden = np.asarray(next(iter(pairs['records'].values()))['values']).shape[-1]
    alignment = {**alignment, 'hidden_size': hidden}
    if sum(np.asarray(row['values']).nbytes for row in pairs['records'].values()) > limits.max_activation_bytes:
        raise ValueError('Paired activations exceed the explicit preparation byte budget')
    parameter_bytes = (4 * hidden * config.width + config.width + 2 * hidden) * 4
    if parameter_bytes * 4 > limits.max_parameter_bytes:
        raise ValueError('Dictionary parameters/gradients/Adam state exceed max_parameter_bytes')
    if parameter_bytes * 3 > limits.max_output_bytes:
        raise ValueError('Safe fitting checkpoint exceeds max_output_bytes estimate')
    runtime = verify_runtime(execution.runtime_profile) if verify_environment else 'explicit-test-runtime'
    import hashlib
    from pathlib import Path
    implementation = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ('fitting.py', 'alignment.py', 'reference.py', 'settings.py', 'fit_worker.py')}
    identity = content_hash({'implementation': implementation, 'alignment': alignment, 'config': asdict(config),
                             'reference': lock(), 'runtime': runtime, 'execution': asdict(execution),
                             'policy': observed_policy(execution), 'sampler': 'seeded-uniform-token-with-replacement-v1'})
    if checkpoint is not None:
        validate_checkpoint(checkpoint, identity)
    limit = config.steps if stop_after is None else stop_after
    if type(limit) is not int or not 0 < limit <= config.steps or (checkpoint is not None and limit < checkpoint['step']):
        raise ValueError('Invalid fit stopping point')
    keys = [row['key'] for row in alignment['rows'] if row['split'] == 'train']
    lengths = [len(pairs['records'][key]['values']) for key in keys]
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    total = int(offsets[-1])
    if not total or total > limits.max_total_tokens:
        raise ValueError('Training tokens exceed the explicit preparation budget')
    if config.batch_size * 2 * hidden * 4 > limits.max_activation_bytes:
        raise ValueError('Fitting batch exceeds activation budget')
    prior_python = random.getstate(); prior_numpy = np.random.get_state()
    devices = [0] if execution.device == 'cuda' else []
    try:
        with numerical_scope(execution, limits), torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed); random.seed(config.seed); np.random.seed(config.seed)
            if devices:
                torch.cuda.manual_seed(config.seed)
            trainer = trainer_for(reference_path, config, alignment, execution)
            sampler = torch.Generator(device='cpu').manual_seed(config.seed + 17)
            losses = []; start = 0
            if checkpoint is not None:
                trainer.ae.load_state_dict(unpack(checkpoint['weights']))
                trainer.optimizer.load_state_dict(unpack(checkpoint['optimizer']))
                trainer.scheduler.load_state_dict(unpack(checkpoint['scheduler']))
                sampler.set_state(torch.from_numpy(checkpoint['sampler_rng']).clone())
                torch.set_rng_state(torch.from_numpy(checkpoint['torch_rng']).clone())
                if devices:
                    if len(checkpoint['cuda_rng']) != 1: raise ValueError('Missing single-device CUDA checkpoint state')
                    torch.cuda.set_rng_state(torch.from_numpy(checkpoint['cuda_rng'][0]).clone(), 0)
                random.setstate(unpack(checkpoint['python_rng'])); np.random.set_state(unpack(checkpoint['numpy_rng']))
                start = checkpoint['step']; losses = list(checkpoint['losses'])
            def snapshot(step):
                result = {'kind':'crosscoder_tensors', 'schema_version':1, 'fit_identity':identity, 'alignment':alignment,
                    'config':asdict(config), 'reference':lock(), 'runtime_identity':runtime, 'execution':asdict(execution),
                    'numerics':observed_policy(execution), 'step':step, 'complete':step == config.steps,
                    'weights':pack(trainer.ae.state_dict()), 'optimizer':pack(trainer.optimizer.state_dict()),
                    'scheduler':pack(trainer.scheduler.state_dict()), 'sampler_rng':sampler.get_state().numpy().copy(),
                    'torch_rng':torch.get_rng_state().numpy().copy(), 'python_rng':pack(random.getstate()),
                    'numpy_rng':pack(np.random.get_state()),
                    'cuda_rng':[torch.cuda.get_rng_state(0).cpu().numpy().copy()] if devices else [],
                    'device':execution.device, 'losses':list(losses)}
                result['checkpoint_digest'] = content_hash(tree_digest(result))
                return result
            for index in range(start, limit):
                check_deadline(deadline)
                selected = torch.randint(total, (config.batch_size,), generator=sampler).numpy()
                batch = torch.from_numpy(training_batch(pairs, keys, offsets, selected)).to(execution.device)
                value = float(trainer.loss(batch.clone()).detach())
                if not np.isfinite(value): raise ValueError('Crosscoder loss is nonfinite')
                losses.append(value); trainer.update(index, batch)
                if on_checkpoint is not None and ((index + 1) % 5 == 0 or index + 1 == limit):
                    on_checkpoint(snapshot(index + 1))
            return snapshot(limit)
    finally:
        random.setstate(prior_python); np.random.set_state(prior_numpy)


class CrosscoderDictionary:
    def __init__(self,checkpoint,reference_path,execution=None):
        import torch
        validate_checkpoint(checkpoint)
        self.execution=execution or Execution()
        if not checkpoint['complete']:raise ValueError('Incomplete fitting checkpoint is not a final dictionary')
        with torch.random.fork_rng(devices=[0] if self.execution.device=='cuda' else []):
            self.trainer=trainer_for(reference_path,FitConfig(**checkpoint['config']),checkpoint['alignment'],self.execution)
        self.model=self.trainer.ae;self.model.load_state_dict(unpack(checkpoint['weights']));self.model.eval();self.model.requires_grad_(False)
        self.checkpoint=checkpoint;self.identity=checkpoint['checkpoint_digest']
    def encode(self,values):
        import torch
        with torch.no_grad():return self.model.encode(torch.from_numpy(np.asarray(values,dtype=np.float32).copy()).to(self.execution.device)).detach().cpu().numpy().copy()
    def decode(self,codes):
        import torch
        with torch.no_grad():return self.model.decode(torch.from_numpy(np.asarray(codes,dtype=np.float32).copy()).to(self.execution.device)).detach().cpu().numpy().copy()
    def feature_delta(self,index,amount):
        if type(index)is not int or not 0<=index<self.checkpoint['config']['width'] or not np.isfinite(amount):raise ValueError('Invalid feature delta')
        codes=np.zeros((1,self.checkpoint['config']['width']),dtype=np.float32);codes[0,index]=amount
        return self.decode(codes)[0]-self.decode(np.zeros_like(codes))[0]
