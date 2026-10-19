"""Explicit offline verification/link step. Never downloads implicitly."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def prepare(cache):
    started = time.monotonic()
    root = Path(__file__).resolve().parent
    lock = json.loads((root / 'model_lock.json').read_text())
    cache = Path(cache).expanduser().resolve(strict=True)
    for item in lock['files']:
        path = cache / item['name']
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise ValueError(f"Incomplete prepared snapshot: {item['name']}")
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                sha.update(chunk)
        if sha.hexdigest() != item['sha256']:
            raise ValueError(f"Pinned snapshot checksum mismatch: {item['name']}")
    destination = root.parent / '.cache/gpt2'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != cache:
            raise ValueError('Existing model link points elsewhere; remove it explicitly before preparing another cache')
    elif destination.exists():
        raise ValueError('Refusing to replace a non-symlink model path')
    else:
        destination.symlink_to(cache, target_is_directory=True)
    return {'revision': lock['revision'], 'cache': str(cache), 'verified_files': len(lock['files']),
            'verification_seconds': time.monotonic() - started, 'download_seconds': None,
            'download_timing_note': 'Verification of already prepared bytes; time download separately.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    print(json.dumps(prepare(parser.parse_args().cache), indent=2))
