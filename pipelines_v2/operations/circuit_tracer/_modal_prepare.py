import json
import modal
from pathlib import Path
app=modal.App('xenon-t20-prepare')
volume=modal.Volume.from_name('xenon-models',create_if_missing=False)
image=modal.Image.debian_slim(python_version='3.12').pip_install('huggingface_hub==0.36.0','hf-xet==1.1.10')
image=image.add_local_file(str(Path(__file__).with_name('prepare.py')),'/opt/prepare.py',copy=True)
@app.function(image=image,cpu=4,memory=8192,timeout=1200,retries=0,max_containers=1,volumes={'/models':volume})
def provision(lock):
    import sys
    sys.path.insert(0,'/opt')
    from prepare import prepare_assets
    result=prepare_assets('/models/circuit-tracer-qwen3',lock)
    volume.commit()
    return result
if __name__=='__main__':
    from pipelines_v2.operations.circuit_tracer.prepare import pair_lock
    with modal.enable_output(),app.run():
        call=provision.spawn(pair_lock())
        try:result=call.get(timeout=1200)
        except BaseException:
            call.cancel(terminate_containers=True);raise
        Path(__import__('os').environ.get('XENON_TRACER_PREPARATION_RECEIPT','prepared-circuit-tracer.json')).write_text(json.dumps({'app_id':app.app_id,'assets':result},indent=2))
        print('Prepared:',app.app_id,{k:v['path'] for k,v in result.items()})
