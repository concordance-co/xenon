"""Explicit SDXL cache preparation and content verification; execution is offline."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def model_lock():
    return json.loads(Path(__file__).with_name('model_lock.json').read_text())


def checkpoint_path(cache_root='/models'):
    lock = model_lock()
    return Path(cache_root) / 'sdxl' / lock['revision']


def verify_checkpoint(path):
    path = Path(path)
    lock = model_lock()
    components = {}
    for name, expected in lock['files'].items():
        item = path / name
        if not item.is_file() or item.stat().st_size != expected['bytes']:
            raise ValueError(f'SDXL component missing or wrong size: {name}; run explicit cache preparation')
        digest = hashlib.sha256()
        with item.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): digest.update(chunk)
        if digest.hexdigest() != expected['sha256']:
            raise ValueError(f'SDXL component checksum mismatch: {name}')
        components.setdefault(name.split('/')[0], {})[name] = expected
    return {'repository':lock['repository'],'revision':lock['revision'],'variant':lock['variant'],
            'components':{name:hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
                          for name,files in components.items()},
            'bytes':sum(item['bytes'] for item in lock['files'].values())}


def prepare_checkpoint(cache_root='/models'):
    """Only explicit CPU preparation downloads assets; no hidden inference download."""
    from huggingface_hub import snapshot_download
    lock = model_lock()
    path = checkpoint_path(cache_root)
    snapshot_download(lock['repository'], revision=lock['revision'], local_dir=path,
                      allow_patterns=list(lock['files']), max_workers=4)
    return verify_checkpoint(path)
