"""Native tensor codec metadata and strict chunk merge identity."""
from copy import deepcopy


def merge_native_payloads(left,right):
    if {k:v for k,v in left.items() if k!='layers'} != {k:v for k,v in right.items() if k!='layers'}:
        raise ValueError('Native tensor chunk identity mismatch')
    result=deepcopy(left)
    for layer,rows in right['layers'].items():
        existing=result['layers'].setdefault(layer,{})
        if set(existing)&set(rows):raise ValueError('Native tensor chunks overlap example coverage')
        existing.update(rows)
    return result


def validate_native_payload(payload):
    import numpy as np
    from pipelines_v2.operations.common.native import NativeSite,HEAD_QUANTITY
    coordinate=NativeSite.from_dict(payload['coordinate'])
    if coordinate.read_or_write!='read' or payload.get('kind')!='native_tensor':
        raise ValueError('Native feature requires a read coordinate')
    if set(payload['layers'])!={str(coordinate.layer)}:
        raise ValueError('Native feature layer mismatch')
    units=list(coordinate.selection.heads or coordinate.selection.neurons)
    for row in payload['layers'][str(coordinate.layer)].values():
        if row.get('units')!=units or row.get('tokens')!=coordinate.tokens.resolve(row['token_count']):
            raise ValueError('Native selected coordinate coverage mismatch')
        values=np.asarray(row['values'])
        shape=(len(row['tokens']),len(units))
        if coordinate.quantity==HEAD_QUANTITY:
            dim=row.get('head_dim')
            if type(dim) is not int or dim<=0:raise ValueError('Native head width missing')
            shape+= (dim,)
        if values.shape!=shape or values.dtype!=np.float32 or not np.isfinite(values).all():
            raise ValueError('Native tensor shape/dtype/finite values mismatch')


def encode_native_payload(payload,*,tensor_key_prefix=''):
    from pipelines_v2.storage.features import encode_tensor_feature_payload
    validate_native_payload(payload)
    return encode_tensor_feature_payload(payload,tensor_key_prefix=tensor_key_prefix)


def decode_native_payload(metadata,tensors):
    from pipelines_v2.storage.features import decode_feature_payload
    payload=decode_feature_payload(metadata,tensors)
    validate_native_payload(payload)
    return payload
