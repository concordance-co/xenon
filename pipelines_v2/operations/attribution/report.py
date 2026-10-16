"""Reduce official producer output to a bounded, cataloged evidence summary."""
from .contracts import validate_result


def summarize_attribution(attribution):
    """Run beside the source artifact, then transfer only this compact result."""
    result = validate_result(attribution.result())
    summary = {key:value for key,value in result.items() if key != 'raw_outputs'}
    summary['raw_outputs'] = {name:{key:value for key,value in entry.items() if key != 'data'}
                              for name,entry in result['raw_outputs'].items()}
    summary['source_artifact_id'] = attribution.id
    summary['source_scientific_identity'] = attribution.manifest().metadata.get('provenance_v1', {}).get('scientific_identity')
    summary['raw_output_location'] = 'Source artifact result.raw_outputs; inventory only in this view'
    return {'payload':summary}
