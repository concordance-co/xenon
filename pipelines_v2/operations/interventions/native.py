"""Explicit selected-coordinate native replacement/scaling; no donor inference."""
from dataclasses import dataclass, replace
from typing import ClassVar, Any
from pipelines_v2.core.types import SpecValidationError, to_primitive
from pipelines_v2.operations.common.native import NativeSite
from .sites import InterventionSite
from .recipes import ActivationPatchSpec


@dataclass(frozen=True, slots=True)
class NativeInterventionSite(InterventionSite):
    coordinate: NativeSite | None = None
    schema_version: int = 1
    kind: ClassVar[str] = 'native_intervention_site'

    def __post_init__(self):
        InterventionSite.__post_init__(self)
        if type(self.schema_version) is not int or self.schema_version != 1 or self.coordinate is None or self.coordinate.read_or_write != 'write':
            raise SpecValidationError('Native intervention needs a versioned write coordinate')
        if self.site != self.coordinate.quantity or self.layers != (self.coordinate.layer,):
            raise SpecValidationError('Native intervention site/coordinate mismatch')

    @classmethod
    def from_capture(cls, site):
        coordinate = replace(site, read_or_write='write')
        return cls(site=coordinate.quantity,layers=coordinate.layers,coordinate=coordinate)

    @classmethod
    def from_dict(cls,p):
        return cls(site=p['site'],layers=tuple(p['layers']),coordinate=NativeSite.from_dict(p['coordinate']),schema_version=p.get('schema_version',1))


@dataclass(frozen=True, slots=True)
class NativeCoordinatePatch(ActivationPatchSpec):
    mode: str = 'scale'
    value: Any = 1.0
    schema_version: int = 1
    kind: ClassVar[str] = 'native_coordinate_patch'
    operator_name: ClassVar[str] = 'native_coordinate'

    def __post_init__(self):
        import json
        ActivationPatchSpec.__post_init__(self)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise SpecValidationError('Unsupported native patch version')
        if not isinstance(self.write_site, NativeInterventionSite) or self.mode not in ('scale','replace'):
            raise SpecValidationError('Native patch requires native coordinate and scale/replace mode')
        if self.strength != 1 or self.source_layer_map or self.application.kind != 'static' or self.application.include_decode or not self.application.include_prompt:
            raise SpecValidationError('Native v1 supports direct static prompt edits only')
        if self.target_tokens != self.write_site.coordinate.tokens:
            raise SpecValidationError('Native patch target tokens must equal its typed coordinate')
        def numeric(value):
            import math
            if isinstance(value,(list,tuple)):
                return bool(value) and all(numeric(v) for v in value)
            return type(value) in (int,float) and math.isfinite(value)
        if not numeric(self.value) or (self.mode == 'scale' and type(self.value) not in (int,float)):
            raise SpecValidationError('Native patch value must be finite numeric values; scale is scalar')
        json.dumps(self.value, allow_nan=False)

    @classmethod
    def from_dict(cls,p):
        from .recipes import patch_application_from_payload
        from pipelines_v2.operations.common.tokens import TokenSelector
        return cls(write_site=NativeInterventionSite.from_dict(p['write_site']),target_tokens=TokenSelector.from_dict(p['target_tokens']),
            application=patch_application_from_payload(p.get('application')),source_layer_map=p.get('source_layer_map',{}),strength=p.get('strength',1),mode=p['mode'],value=p['value'],schema_version=p.get('schema_version',1))
