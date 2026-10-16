"""Pinned standard ReLU dictionary with explicit original-GPT2 coordinate map."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np

REPOSITORY = 'jbloom/GPT2-Small-SAEs-Reformatted'
REVISION = '57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9'
WEIGHTS_SHA256 = '3ff899406a9fb3c33aebca365d6a8065a2f48eba9cef699543fb8602abe207be'
CONFIG_SHA256 = 'ac362df90ad640a8e78def8d4f595cbc7c67c049f06698ca3b5cc5a9f5adb873'
MODEL_REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
SITE = 'blocks.6.hook_resid_pre'
ADAPTER_VERSION = 'gpt2-jb-centered-standard-v1'


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def centered(value):
    return value - value.mean(axis=-1, keepdims=True)


@dataclass(frozen=True)
class SAEDictionary:
    """NumPy FP32 inference; never loads pickle or imports the training stack.

    HF residual -> per-token mean subtraction -> dictionary coordinates.
    Decoder results are centered before lifting; original mean is preserved.
    The all-ones residual component is a GPT2 LayerNorm gauge, not an SAE feature.
    """
    weights: dict
    metadata: dict

    def __post_init__(self):
        required = {'W_enc', 'W_dec', 'b_enc', 'b_dec'}
        if set(self.weights) != required: raise ValueError('Dictionary needs exactly encoder/decoder and both biases')
        enc, dec = self.weights['W_enc'], self.weights['W_dec']
        if enc.ndim != 2 or dec.shape != enc.T.shape: raise ValueError('Encoder/decoder dimensions disagree')
        if self.weights['b_enc'].shape != (enc.shape[1],) or self.weights['b_dec'].shape != (enc.shape[0],):
            raise ValueError('Dictionary bias dimensions disagree')
        if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in self.weights.values()):
            raise ValueError('Dictionary requires finite FP32 weights')
        if self.metadata.get('architecture') != 'standard' or self.metadata.get('activation') != 'relu' or self.metadata.get('normalization') != 'none':
            raise ValueError('Only standard ReLU with no activation scaling is supported')

    @property
    def d_in(self): return self.weights['W_enc'].shape[0]
    @property
    def d_sae(self): return self.weights['W_enc'].shape[1]

    @classmethod
    def import_pinned(cls, cache_root, *, model_checkpoint_digest, coordinate_evidence):
        from safetensors.numpy import load_file
        root = Path(cache_root)
        for name, expected in [('cfg.json', CONFIG_SHA256), ('sae_weights.safetensors', WEIGHTS_SHA256)]:
            if file_digest(root/name) != expected: raise ValueError(f'Pinned SAE {name} checksum mismatch')
        cfg = json.loads((root/'cfg.json').read_text())
        if (cfg['model_name'], cfg['hook_point'], cfg['d_in'], cfg['d_sae']) != ('gpt2-small', SITE, 768, 24576):
            raise ValueError('SAE model/site/dimensions mismatch')
        validate_coordinate_evidence(coordinate_evidence, model_checkpoint_digest)
        metadata = {'schema_version': 1, 'type': 'sae', 'adapter_version': ADAPTER_VERSION,
            'repository': REPOSITORY, 'revision': REVISION, 'weights_sha256': WEIGHTS_SHA256,
            'config_sha256': CONFIG_SHA256, 'architecture': 'standard', 'activation': 'relu',
            'normalization': 'none', 'apply_b_dec_to_input': True, 'decoder_convention': 'features @ W_dec + b_dec',
            'input_site': {'model': 'openai-community/gpt2', 'revision': MODEL_REVISION,
                'checkpoint_digest': model_checkpoint_digest, 'site': 'resid_pre', 'layer': 6, 'dimension': 768},
            'coordinate_map': 'subtract per-token channel mean; lift with original mean',
            'model_loading_transformations': {'center_writing_weights': True, 'fold_ln': True,
                'center_unembed': True, 'fold_value_biases': True, 'refactor_factored_attn_matrices': False},
            'training': {'dataset': cfg['dataset_path'], 'tokens': cfg['total_training_tokens'],
                'seed': cfg['seed'], 'context_size': cfg['context_size'], 'exact_training_model_revision': None},
            'coordinate_evidence': coordinate_evidence}
        return cls(load_file(str(root/'sae_weights.safetensors')), metadata)

    def encode(self, coordinates):
        x = np.asarray(coordinates, dtype=np.float32)
        if x.shape[-1] != self.d_in or not np.isfinite(x).all(): raise ValueError('Invalid SAE input')
        return np.maximum((x-self.weights['b_dec']) @ self.weights['W_enc']+self.weights['b_enc'], 0)

    def decode(self, features):
        z = np.asarray(features, dtype=np.float32)
        if z.shape[-1] != self.d_sae or not np.isfinite(z).all(): raise ValueError('Invalid SAE features')
        return z @ self.weights['W_dec'] + self.weights['b_dec']

    def encode_original(self, original): return self.encode(centered(np.asarray(original, dtype=np.float32)))

    def fidelity(self, original):
        x = centered(np.asarray(original, dtype=np.float32)); z = self.encode(x); decoded = self.decode(z)
        error = np.mean((x-decoded)**2); variance = np.var(x.reshape(-1, self.d_in), axis=0).mean()
        return {'mse': float(error), 'normalized_mse': float(error/max(float(np.mean(x*x)), 1e-30)),
            'mean_l0': float(np.count_nonzero(z, axis=-1).mean()),
            'explained_variance': float(1-error/variance) if variance > 0 else None,
            'decoder_mean_rms': float(np.sqrt(np.mean(decoded.mean(-1)**2)))}


def validate_coordinate_evidence(evidence, checkpoint_digest):
    if len(checkpoint_digest) != 64 or any(c not in '0123456789abcdef' for c in checkpoint_digest):
        raise ValueError('Require the actual original model checkpoint digest')
    expected = {'adapter_version': ADAPTER_VERSION, 'model_checkpoint_digest': checkpoint_digest,
        'model_revision': MODEL_REVISION, 'dictionary_weights_sha256': WEIGHTS_SHA256,
        'site': SITE, 'coordinate_map': 'per_token_center', 'passed': True}
    if any(evidence.get(k) != v for k, v in expected.items()): raise ValueError('Coordinate evidence does not match dictionary/model/site')
    if evidence.get('prompt_count', 0) < 2 or not evidence.get('reference_versions'):
        raise ValueError('Coordinate evidence must include paired real-model reference observations')
    for key in ('tensor_max_abs', 'encode_max_abs', 'decode_max_abs'):
        value = evidence.get(key)
        if not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0 or value > 0.005:
            raise ValueError('Coordinate/SAE reference error exceeds fixed 0.005 ceiling')


def verify_original_model(path):
    expected = {
        'model.safetensors': '248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707',
        'config.json': '0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4',
        'tokenizer.json': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
        'vocab.json': '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783',
        'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
    }
    for name,digest in expected.items():
        if file_digest(Path(path)/name)!=digest:raise ValueError(f'Original GPT2 {name} differs from pinned checkpoint')
    from pipelines_v2.engine.torch.engine import checkpoint_identity
    return checkpoint_identity(path)
