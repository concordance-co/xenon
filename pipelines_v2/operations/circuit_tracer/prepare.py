"""Explicit CPU-only preparation. Normal tracing never downloads model assets."""
import hashlib
import json
from pathlib import Path


def pair_lock():return json.loads(Path(__file__).with_name('pair_lock.json').read_text())


def verify_files(root, files):
    observed={}
    for name,expected in files.items():
        path=Path(root)/name
        if path.stat().st_size!=expected['bytes']:raise ValueError(f'Prepared asset size mismatch: {name}')
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
        if h.hexdigest()!=expected['sha256']:raise ValueError(f'Prepared asset digest mismatch: {name}')
        observed[name]={'sha256':h.hexdigest(),'bytes':path.stat().st_size}
    return observed


def prepare_assets(root, lock):
    """Run only during explicit provision, outside any GPU function."""
    import subprocess
    result={}
    for role in ('model','dictionary'):
        item=lock[role];path=Path(root)/role/item['revision'];path.mkdir(parents=True,exist_ok=True)
        missing=[name for name,entry in item['files'].items() if not (path/name).is_file() or (path/name).stat().st_size!=entry['bytes']]
        if missing:
            subprocess.run(['hf','download',item['repository'],*missing,'--revision',item['revision'],'--local-dir',str(path),'--max-workers','4'],check=True,timeout=1000)
        result[role]={'path':str(path),'repository':item['repository'],'revision':item['revision'],'files':verify_files(path,item['files'])}
    (Path(root)/'prepared.json').write_text(json.dumps(result,indent=2))
    return result
