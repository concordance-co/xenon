"""Explicit crosscoder execution policies and finite workload budgets."""
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import math
import time


@dataclass(frozen=True)
class Execution:
    device: str = 'cpu'
    dtype: str = 'float32'
    runtime_profile: str = 'darwin-arm64'
    threads: int = 2

    def __post_init__(self):
        if self.device not in ('cpu', 'cuda') or self.dtype != 'float32':
            raise ValueError('Crosscoder supports explicit CPU/CUDA float32 eager execution')
        if self.runtime_profile not in ('darwin-arm64', 'linux-x86_64-cuda'):
            raise ValueError('Unsupported crosscoder runtime profile')
        if self.device == 'cuda' and self.runtime_profile != 'linux-x86_64-cuda':
            raise ValueError('CUDA execution requires the Linux CUDA runtime lock')
        if type(self.threads) is not int or self.threads < 1:
            raise ValueError('threads must be a positive integer')


@dataclass(frozen=True)
class Limits:
    max_examples: int = 1024
    max_tokens_per_example: int = 512
    max_total_tokens: int = 32768
    max_activation_bytes: int = 256 * 1024**2
    max_parameter_bytes: int = 512 * 1024**2
    max_output_bytes: int = 512 * 1024**2
    max_process_rss_bytes: int = 8 * 1024**3
    max_device_bytes: int = 16 * 1024**3
    timeout_seconds: float = 600

    def __post_init__(self):
        for name, value in asdict(self).items():
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0 if name == 'timeout_seconds' else type(value) is int and value > 0
            if not valid:
                raise ValueError(f'{name} must be finite and positive')

    def deadline(self):
        return time.monotonic() + self.timeout_seconds


def check_deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError('Crosscoder operation exceeded its explicit wall-time budget')


def observed_policy(execution):
    import torch
    policy = {'dtype': execution.dtype, 'attention_implementation': 'eager',
              'device': execution.device, 'determinism': 'deterministic_algorithms',
              'compile': False, 'matmul': 'highest; tf32_disabled',
              'threads': execution.threads, 'policy_version': 'crosscoder-fp32-v2'}
    if execution.device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA crosscoder requires an observed CUDA device')
        policy['gpu'] = {'name': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
                         'total_memory': torch.cuda.get_device_properties(0).total_memory, 'cuda': torch.version.cuda}
    return policy


@contextmanager
def numerical_scope(execution, limits=None):
    import torch
    observed_policy(execution)
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.set_num_threads(execution.threads)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    prior_fraction = None
    if execution.device == 'cuda' and limits is not None:
        prior_fraction = torch.cuda.get_per_process_memory_fraction(0)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(limits.max_device_bytes / total, 1.0), 0)
    try:
        yield observed_policy(execution)
    finally:
        if prior_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(prior_fraction, 0)
        torch.set_num_threads(previous[0]); torch.use_deterministic_algorithms(previous[1])
        torch.set_float32_matmul_precision(previous[2]); torch.backends.cuda.matmul.allow_tf32 = previous[3]
        torch.backends.cudnn.allow_tf32 = previous[4]; torch.backends.cudnn.deterministic = previous[5]; torch.backends.cudnn.benchmark = previous[6]
