"""Explicit CPU-only preparation of pinned crosscoder assets on an existing Modal volume."""
import argparse
import json


def prepare(volume_name='xenon-models'):
    import modal
    from pathlib import Path
    from .reference import runtime_spec
    from pipelines_v2.runtime.locks import build_locked_image
    runtime=runtime_spec('linux-x86_64-cuda')
    image=build_locked_image(modal,runtime).add_local_dir(str(Path(__file__).resolve().parents[2]),'/root/pipelines_v2')
    volume=modal.Volume.from_name(volume_name,create_if_missing=False)
    app=modal.App('xenon-crosscoder-prepare')

    @app.function(image=image,volumes={'/models':volume},cpu=4,memory=8192,timeout=1200,
                  max_containers=1,serialized=True)
    def worker():
        import subprocess,sys
        from pipelines_v2.representation.crosscoder.reference import verify_runtime
        from pipelines_v2.representation.crosscoder.prepare import prepare_assets
        verify_runtime('linux-x86_64-cuda')
        subprocess.run([sys.executable,'-m','pip','check'],check=True,timeout=60)
        result=prepare_assets('/models/xenon-crosscoder/reference','/models/hub','/models/xenon-crosscoder/models.json')
        volume.commit()
        return result

    with modal.enable_output(),app.run():
        result=worker.remote()
        return {'app_id':app.app_id,**result}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--volume',default='xenon-models')
    args=parser.parse_args();print(json.dumps(prepare(args.volume),indent=2))


if __name__=='__main__':main()
