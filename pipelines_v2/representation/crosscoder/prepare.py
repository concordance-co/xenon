"""Explicit online CPU preparation; actual workflows remain offline."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
from .reference import lock,reference_modules
from .qwen import REPOSITORIES,REVISIONS,model_identity


def prepare_assets(reference,cache,models):
    from huggingface_hub import snapshot_download
    root=Path(reference);manifest=lock()
    for relative,digest in manifest['files'].items():
        path=root/relative
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:continue
        url=f"https://raw.githubusercontent.com/science-of-finetuning/crosscoder_learning/{manifest['revision']}/{relative}"
        with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Prepared reference bytes differ: '+relative)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    reference_modules(root)
    model_lock=json.loads(Path(__file__).with_name('model_lock.json').read_text())
    prepared=[]
    for side,(repository,revision) in enumerate(zip(REPOSITORIES,REVISIONS)):
        path=snapshot_download(repository,revision=revision,cache_dir=cache,allow_patterns=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','model.safetensors','LICENSE'])
        identity=model_identity(path,side)
        prepared.append({'repository':repository,'revision':revision,'path':path,'identity':identity})
    target=Path(models);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps({'models':prepared},indent=2)+'\n')
    return {'manifest':str(target),'models':prepared,'reference':str(root),'reference_revision':manifest['revision']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference',required=True);parser.add_argument('--cache',required=True);parser.add_argument('--models',required=True)
    args=parser.parse_args();print(json.dumps(prepare_assets(args.reference,args.cache,args.models),indent=2))

if __name__=='__main__':main()
