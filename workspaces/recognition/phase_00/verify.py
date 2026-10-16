"""Independent direct-model, preprocessing and selected-coordinate oracle; offline."""
import argparse,json,os
from dataclasses import replace
from pathlib import Path
from pipelines_v2.engine.recognition import ViTRecognitionEngine, ViTSite
from pipelines_v2.data.images import ImageInput


def verify(model, inputs):
    import numpy as np
    import torch
    from PIL import Image
    image=ImageInput.from_dict(json.loads((Path(inputs)/'inputs.json').read_text())[0])
    site=ViTSite(tokens=(0,1,98),channels=tuple(range(8)))
    edit=replace(site,tokens=(98,),read_or_write='write')
    with ViTRecognitionEngine(model).open_session() as session:
        pixels=session.preprocess(image,roots=(inputs,))
        with Image.open(image.store_ref['path']) as original:
            expected=(np.asarray(original.resize((224,224),resample=Image.Resampling.BILINEAR)).astype('float32')/255.-.5)/.5
        processor_error=float(np.max(np.abs(pixels[0].numpy()-expected.transpose(2,0,1))))
        assert processor_error<=3e-7
        baseline,values=session.forward(pixels,site)
        block=session.model.vit.encoder.layer[0];observed=[]
        handle=block.register_forward_hook(lambda m,a,o:observed.append(o.detach().clone()))
        try:
            with torch.no_grad(): direct=session.model(pixel_values=pixels).logits
        finally:handle.remove()
        capture_error=float((values-observed[0][:,list(site.tokens)][:,:,list(site.channels)]).abs().max())
        direct_error=float((baseline-direct).abs().max())
        identity,_=session.forward(pixels,site,edit=edit,scale=1.)
        altered,altered_values=session.forward(pixels,site,edit=edit,scale=0.)
        before_after=[]
        def independent_hook(m,a,o):
            mask=torch.ones_like(o);mask[:,98,:8]=0
            patched=o*mask;before_after.append((o.detach().clone(),patched.detach().clone()))
            return patched
        handle=block.register_forward_hook(independent_hook)
        try:
            with torch.no_grad(): direct_edit=session.model(pixel_values=pixels).logits
        finally:handle.remove()
        unselected=torch.ones_like(before_after[0][0],dtype=torch.bool);unselected[:,98,:8]=False
        outside_error=float((before_after[0][0]-before_after[0][1])[unselected].abs().max())
        edit_error=float((altered-direct_edit).abs().max())
        assert direct_error==capture_error==edit_error==outside_error==0
        assert torch.equal(identity,baseline) and torch.count_nonzero(altered_values[:,2,:])==0
        assert not block._forward_hooks
        result={'processor_max_error':processor_error,'direct_forward_error':direct_error,'capture_error':capture_error,
            'independent_edit_error':edit_error,'unselected_coordinate_error':outside_error,'identity_exact':True,
            'hooks_cleaned':True,'max_logit_change':float((altered-baseline).abs().max()),
            'scope':'one synthetic-image engineering oracle; no scientific evaluation'}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=verify(a.model,a.inputs);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
