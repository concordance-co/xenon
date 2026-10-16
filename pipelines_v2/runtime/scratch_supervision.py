"""Process-tree supervision and strict OS isolation for opt-in local scratch."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


class ScratchLimitError(RuntimeError):
    """Execution exceeded a declared scratch envelope; no result may publish."""


class ScratchPreparationError(RuntimeError):
    """Required offline tooling/cache is absent; prepare it explicitly first."""


@dataclass(frozen=True, slots=True)
class ScratchLimits:
    max_examples: int = 64
    max_input_tokens: int = 256
    max_generated_tokens: int = 32
    max_artifact_bytes: int = 256 * 1024 * 1024
    timeout_seconds: float = 600
    max_process_tree_rss_bytes: int = 8 * 1024 * 1024 * 1024

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == 'timeout_seconds':
                valid = type(value) in (int, float) and math.isfinite(value) and value > 0
            else:
                valid = type(value) is int and value > 0
            if not valid:
                raise ValueError(f'Scratch limit {key} must be finite and positive')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


def require_psutil():
    try:
        import psutil
    except ImportError as exc:
        raise ScratchPreparationError('local-scratch requires psutil; install the locked Xenon dependencies before running') from exc
    return psutil


def tree_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ScratchLimitError('Scratch outputs cannot contain symlinks')
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # An atomic writer may have renamed its temporary file.
    return total


def offline_command(command: Sequence[str], *, writable_root: Path) -> list[str]:
    """No fallback to an unenforced network policy on unsupported hosts."""
    writable = str(writable_root.resolve())
    if sys.platform == 'darwin':
        sandbox = shutil.which('sandbox-exec')
        if sandbox is None:
            raise ScratchPreparationError('Strict offline scratch requires macOS sandbox-exec')
        # JSON quoted strings have the escaping required by the sandbox profile.
        policy = '(version 1)(allow default)(deny network*)(deny file-write*)' + \
                 f'(allow file-write* (subpath {json.dumps(writable)}) (literal "/dev/null"))'
        return [sandbox, '-p', policy, *command]
    if sys.platform.startswith('linux'):
        bwrap = shutil.which('bwrap')
        if bwrap is None:
            raise ScratchPreparationError('Strict offline scratch requires bubblewrap (bwrap) with unprivileged namespaces; install it explicitly before running')
        return [bwrap, '--unshare-net', '--die-with-parent', '--ro-bind', '/', '/',
                '--bind', writable, writable, '--dev-bind', '/dev', '/dev', '--proc', '/proc', '--', *command]
    raise ScratchPreparationError(f'Strict offline scratch has no supported OS sandbox on {sys.platform}')


def scratch_environment(attempt: Path, *, inherited: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if inherited is None else inherited)
    guard = Path(__file__).parent / '_scratch_guard'
    package_root = Path(__file__).resolve().parents[2]
    env['PYTHONPATH'] = os.pathsep.join([str(guard), str(package_root), env.get('PYTHONPATH', '')])
    env.update(XENON_LOCAL_SCRATCH='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               MPLCONFIGDIR=str(attempt / 'cache' / 'matplotlib'), TMPDIR=str(attempt / 'tmp'),
               XDG_CACHE_HOME=str(attempt / 'cache'), XENON_HOME=str(attempt / 'state'))
    for name in ('tmp', 'cache', 'state'):
        (attempt / name).mkdir(parents=True, exist_ok=True)
    return env


def supervise(command: Sequence[str], *, attempt: Path, output_root: Path, limits: ScratchLimits,
              env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Measure total descendant RSS, enforce wall time/output growth, kill tree.

    RSS is observed every 20ms; it is not a device-memory reservation or a
    claim about unobserved transient peaks. All artifact writes have a second,
    synchronous store guard, and publication follows an exact final size check.
    """
    psutil = require_psutil()
    attempt.mkdir(parents=True, exist_ok=True)
    sandboxed = offline_command(command, writable_root=attempt)
    started = time.monotonic()
    peak_rss = 0
    known = {}
    process = None
    error = None
    stdout = attempt / 'stdout.log'
    stderr = attempt / 'stderr.log'
    diagnostics = {'schema_version':1, 'limits':limits.to_dict(), 'status':'running',
                   'network_policy':'OS sandbox + inherited Python sentinel', 'rss_sample_interval_seconds':0.02}
    try:
        with stdout.open('wb') as out, stderr.open('wb') as err:
            process = subprocess.Popen(sandboxed, stdout=out, stderr=err, env=scratch_environment(attempt, inherited=env), start_new_session=True)
            root = psutil.Process(process.pid)
            while True:
                elapsed = time.monotonic() - started
                if elapsed > limits.timeout_seconds:
                    raise ScratchLimitError('Scratch wall-time limit exceeded')
                try:
                    live = [root, *root.children(recursive=True)]
                except psutil.NoSuchProcess:
                    live = []
                for child in live:
                    known[child.pid] = child
                rss = 0
                for child in list(known.values()):
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak_rss = max(peak_rss, rss)
                if rss > limits.max_process_tree_rss_bytes:
                    raise ScratchLimitError('Scratch aggregate process-tree RSS limit exceeded')
                if tree_bytes(output_root) > limits.max_artifact_bytes:
                    raise ScratchLimitError('Scratch artifact-byte limit exceeded')
                # Bound logs and other side effects as well; they cannot grow forever.
                if tree_bytes(attempt) > limits.max_artifact_bytes + 16 * 1024 * 1024:
                    raise ScratchLimitError('Scratch auxiliary-output byte limit exceeded')
                code = process.poll()
                if code is not None:
                    if code != 0:
                        detail = stderr.read_text(errors='replace')[-4000:]
                        raise RuntimeError(f'Scratch worker failed (exit {code}): {detail}')
                    remaining = [child for child in known.values() if child.pid != process.pid and child.is_running() and child.status() != psutil.STATUS_ZOMBIE]
                    if remaining:
                        raise ScratchLimitError('Scratch worker exited with unfinished child processes')
                    if tree_bytes(output_root) > limits.max_artifact_bytes:
                        raise ScratchLimitError('Scratch final artifact-byte limit exceeded')
                    diagnostics['status'] = 'passed'
                    break
                time.sleep(0.02)
    except BaseException as exc:
        error = exc
        diagnostics['status'] = 'failed'
        diagnostics['error'] = str(exc)
    finally:
        if process is not None:
            # Kill the entire process group, including native children that outlive
            # the Python parent. Previously observed reparented children are killed too.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            for child in known.values():
                try:
                    if child.is_running():
                        child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(list(known.values()), timeout=0.25)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        diagnostics.update(elapsed_seconds=time.monotonic()-started, observed_peak_tree_rss_bytes=peak_rss,
                           artifact_bytes=tree_bytes(output_root))
        (attempt / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    if error is not None:
        raise error
    return diagnostics
