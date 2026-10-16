from pipelines_v2.evidence import compare_evidence


def test_sparse_legacy_record_cannot_claim_c05_compatibility():
    record = {'metric':'margin', 'evaluator_digest':'not-a-digest', 'split':'heldout',
              'pack_identity':'not-a-pack', 'evidence_level':'engineering', 'effect':{'value':1.2}}
    artifact = {'artifact_id':'a', 'identity_complete':True, 'provenance':{}, 'result':{'records':[record]}}
    result = compare_evidence([artifact, {**artifact, 'artifact_id':'b'}], metric='margin')
    assert not result['comparable'] and result['aggregation'] == 'none'
    assert all('unverified' in row['reasons'][0] for row in result['rows'])
