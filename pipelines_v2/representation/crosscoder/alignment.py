"""Explicit ordered model-side, document and token coordinates; no row truncation."""
import numpy as np
from pipelines_v2.core.provenance import content_hash
from pipelines_v2.storage.image_features import tensor_content_digest


def align_sides(sides, models, *, layer, train_keys, execution=None):
    if len(sides)!=2 or len(models)!=2 or models[0]['checkpoint_digest']==models[1]['checkpoint_digest']:
        raise ValueError('Two distinct ordered model sides required')
    keys=[row['key'] for row in sides[0]]
    if not keys or len(keys)!=len(set(keys)) or keys!=[row['key'] for row in sides[1]]:
        raise ValueError('Missing, duplicate or swapped paired document rows')
    if not train_keys or len(train_keys)!=len(set(train_keys)) or not set(train_keys)<set(keys):
        raise ValueError('Explicit disjoint train/heldout document split required')
    records={}
    for left,right in zip(*sides,strict=True):
        ids=left['token_ids'];positions=left['positions']
        if not ids or ids!=right['token_ids'] or positions!=right['positions'] or positions!=list(range(len(ids))):
            raise ValueError('Token IDs or positions differ across model sides')
        if left['prompt_digest']!=right['prompt_digest']:
            raise ValueError('Paired text identity differs')
        arrays=[np.asarray(row['values'],dtype=np.float32) for row in (left,right)]
        if arrays[0].shape!=arrays[1].shape or arrays[0].ndim!=2 or arrays[0].shape[0]!=len(ids) or not all(np.isfinite(a).all() for a in arrays):
            raise ValueError('Paired activations do not match token/hidden coordinates')
        records[left['key']]={'token_ids':ids,'positions':positions,'prompt_digest':left['prompt_digest'],
            'split':'train' if left['key'] in train_keys else 'heldout','values':np.stack(arrays,axis=1)}
    alignment={'schema_version':1,'model_sides':models,'layer':layer,'site':'resid_pre',
        'adapter_version':'qwen2.5-residual-fp32-v2','execution':execution or {'device':'cpu','dtype':'float32'},'template':'none; raw identical token IDs',
        'rows':[{'key':key,**{name:records[key][name] for name in ('token_ids','positions','prompt_digest','split')},
                'activation_digest':tensor_content_digest(records[key]['values'])} for key in keys]}
    alignment['identity']=content_hash(alignment)
    return {'kind':'crosscoder_tensors','alignment':alignment,'records':records}


def validate_pairs(payload):
    alignment=payload['alignment'];body={k:v for k,v in alignment.items() if k!='identity'}
    if alignment['identity']!=content_hash(body):raise ValueError('Alignment record integrity mismatch')
    if set(payload['records'])!={row['key'] for row in alignment['rows']}:
        raise ValueError('Paired row order changed')
    for row in alignment['rows']:
        record=payload['records'][row['key']]
        if any(record[name]!=row[name] for name in ('token_ids','positions','prompt_digest','split')):
            raise ValueError('Paired coordinate metadata changed')
        if tensor_content_digest(np.asarray(record['values']))!=row['activation_digest']:
            raise ValueError('Paired activation/model-side content changed')
    return alignment
