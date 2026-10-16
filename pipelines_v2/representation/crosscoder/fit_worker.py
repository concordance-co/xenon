"""Private CPU fitting process: JSON/safetensors boundary, no pickle or core runner."""
import json,sys
from pathlib import Path
from pipelines_v2.storage.local import LocalArtifactStore
from pipelines_v2.storage.features import write_capture_features,load_feature_payload
from .fitting import FitConfig,fit
from .settings import Execution,Limits


def main(path):
    root=Path(path);request=json.loads((root/'request.json').read_text());store=LocalArtifactStore(root)
    pairs=load_feature_payload(store,request['pairs'])
    prior=load_feature_payload(store,request['checkpoint']) if request['checkpoint'] is not None else None
    result=fit(pairs,request['reference_path'],FitConfig(**request['config']),checkpoint=prior,stop_after=request['stop_after'],execution=Execution(**request['execution']),limits=Limits(**request['limits']))
    ref=write_capture_features(store,'output',{'checkpoint':result})['checkpoint']
    (root/'response.json').write_text(json.dumps(ref.to_dict() if hasattr(ref,'to_dict') else ref))

if __name__=='__main__':main(sys.argv[1])
