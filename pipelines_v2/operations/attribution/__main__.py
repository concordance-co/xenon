"""Explicit one-time preparation; ordinary planning/execution never downloads assets."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from .specs import official_tool_lock,_verify_files
from .contracts import locked_assets


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare']);p.add_argument('--root',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--model-name', choices=('gpt2','qwen2.5'), default='gpt2')
    a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    assets=locked_assets(a.model_name);_verify_files(a.model,assets['model']['files'])
    lock=official_tool_lock();uv=shutil.which('uv')
    if not uv:raise RuntimeError('Install uv before preparing the isolated upstream environment')
    tool=root/'mattr';mib=root/'mib';venv=root/'venv'
    for url,dest,revision in [(lock['repository'],tool,lock['revision']),(lock['mib_repository'],mib,lock['mib_revision'])]:
        if not dest.exists():
            subprocess.run(['git','clone',url,str(dest)],check=True,timeout=120)
            subprocess.run(['git','-C',str(dest),'checkout','--detach',revision],check=True,timeout=30)
        actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True,timeout=30).strip()
        if actual!=revision:raise RuntimeError(f'Existing checkout has another revision: {dest}')
    subprocess.run(['git','-C',str(mib),'submodule','update','--init','--recursive'],check=True,timeout=120)
    subprocess.run([uv,'sync','--frozen','--no-dev','--python','3.12','--project',str(tool)],env={**os.environ,'UV_PROJECT_ENVIRONMENT':str(venv)},check=True,timeout=900)
    dataset=root/'ioi';dataset.mkdir(exist_ok=True)
    for name,expected in assets['data']['files'].items():
        path=dataset/name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected:continue
        url=f"https://huggingface.co/datasets/mib-bench/ioi/resolve/{assets['data']['revision']}/{name}"
        with urllib.request.urlopen(url,timeout=60) as response:contents=response.read(64*1024**2+1)
        if hashlib.sha256(contents).hexdigest()!=expected:raise RuntimeError('Pinned dataset checksum mismatch')
        temp=path.with_suffix('.partial');temp.write_bytes(contents);temp.replace(path)
    config={'tool_root':str(tool),'mib_root':str(mib),'tool_python':str(venv/'bin/python'),'model_path':str(a.model.resolve()),'dataset_path':str(dataset),'artifact_root':str(root/'artifacts')}
    from pipelines_v2.runtime.external_tool import ExternalToolRuntimeSpec,check_tool_runtime
    runtime=ExternalToolRuntimeSpec(tool_lock=lock,**{k:config[k] for k in ('tool_root','mib_root','tool_python')})
    check_tool_runtime(runtime)
    target=root/'execution.json';target.write_text(json.dumps(config,indent=2)+'\n');print(target)

if __name__=='__main__':main()
