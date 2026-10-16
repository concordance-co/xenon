"""Direct interventions run on the original model; no replacement-flow inference."""
from dataclasses import asdict
from pipelines_v2.core.types import to_primitive
from pipelines_v2.operations.common.native import NativeSite
from pipelines_v2.operations.interventions.native import NativeCoordinatePatch, NativeInterventionSite
from pipelines_v2.operations.execution.common import OperationExecutionResult
from .contracts import AttributionGraph, GraphNode, GraphEdge, RouteEvidence, digest
from .gradients import embedding_candidates


def evaluate_routes(spec, session):
    import torch
    nodes=[GraphNode.create('native_site',to_primitive(site),site.name) for site in spec.sites]
    target={'metric':'next_token_logit_margin', 'definition':'correct minus foil; token IDs resolved per pair', 'position':'last_prompt_token'}
    output=GraphNode.create('output', {'model_identity':session.model_identity,**target}, 'Next-token margin')
    evidence=[]; candidates=[]; examples=[]
    for pair in spec.pairs:
        encoded={arm:session.encode([text]) for arm,text in [('baseline',pair.prompt),('opposite',pair.opposite),('same_label',pair.same_label),('unmatched',pair.unmatched)]}
        if any(ids.shape[1]>spec.max_prompt_tokens for ids,_ in encoded.values()):raise ValueError('Route prompt exceeds declared token ceiling')
        correct=session.tokenizer.encode(pair.correct,add_special_tokens=False);foil=session.tokenizer.encode(pair.foil,add_special_tokens=False)
        if len(correct)!=1 or len(foil)!=1 or correct==foil:raise ValueError('Route target and foil must be distinct single tokens')
        c,f=correct[0],foil[0]
        ids,mask=encoded['baseline']
        baseline,captures=session.forward(ids,mask,capture=spec.sites)
        before=float(baseline[0,-1,c]-baseline[0,-1,f])
        donor_values={}
        for arm in ('opposite','same_label','unmatched'):
            _,donor_values[arm]=session.forward(*encoded[arm],capture=spec.sites)
        examples.append({'key':pair.key,'template':pair.template,'prompt':pair.prompt,'source':pair.source,
                         'target_token_ids':{'correct':c,'foil':f},'baseline_margin':before})
        if spec.ig_steps:
            embeddings=session.model.get_input_embeddings()(ids).detach()
            def scalar(value):
                logits,_=session.forward(attention_mask=mask,inputs_embeds=value,differentiable=torch.is_grad_enabled())
                return logits[0,-1,c]-logits[0,-1,f]
            candidate=embedding_candidates(scalar,embeddings,steps=spec.ig_steps)
            candidates.append({'example_key':pair.key,'token_ids':ids[0].tolist(),'target':{'correct':c,'foil':f},**candidate})
        for site,node in zip(spec.sites,nodes):
            write=NativeInterventionSite.from_capture(site)
            for arm in ('identity','zero','opposite','same_label','unmatched'):
                value=0. if arm=='zero' else (captures if arm=='identity' else donor_values[arm])[site.name][0].tolist()
                patch=NativeCoordinatePatch(write_site=write,target_tokens=site.tokens,mode='replace',value=value)
                observed,_=session.forward(ids,mask,edits=(patch,))
                after=float(observed[0,-1,c]-observed[0,-1,f])
                difference=float((observed-baseline).abs().max())
                p=torch.log_softmax(baseline[0,-1],-1);q=torch.log_softmax(observed[0,-1],-1)
                status='rejected' if arm=='identity' and difference>1e-6 else 'passed'
                evidence.append(RouteEvidence.create(node_id=node.node_id,example_key=pair.key,split='heldout',template=pair.template,
                    arm=arm,read_site=to_primitive(site),write_site=to_primitive(write.coordinate),
                    donor={'role':arm,'prompt':getattr(pair,arm,pair.prompt) if arm!='zero' else None,
                           'token_positions':site.tokens.resolve(encoded[arm][0].shape[1]) if arm in encoded else site.tokens.resolve(ids.shape[1])},
                    intervention={'mode':'replace','target_token_ids':{'correct':c,'foil':f},'value_digest':digest(value),'recipient_positions':site.tokens.resolve(ids.shape[1]),
                                  'scope':'selected last-prompt head output; original downstream model recomputed'},
                    target=target,baseline=before,observed=after,
                    collateral={'max_logit_change':difference,'next_token_kl':max(0.,float((p.exp()*(p-q)).sum()))},engineering_status=status))
    edges=[]
    for node in nodes:
        rows=[e for e in evidence if e.node_id==node.node_id and e.arm=='zero']
        edges.append(GraphEdge(node.node_id,output.node_id,'native_intervention_effect','mean_target_margin_delta','logit',
            sum(e.observed-e.baseline for e in rows)/len(rows),'unmodified original model','zero',tuple(e.evidence_id for e in rows)))
    graph=AttributionGraph(source_run_ids=(f'request:{spec.semantic_hash()}',),provenance={'operation_identity':spec.semantic_hash(),'pair_content_identity':digest([asdict(p) for p in spec.pairs]),'artifact_provenance':'enclosing manifest provenance_v1'},
        target=target,model_identity=session.model_identity,algorithm={'name':'original-model-native-route','version':1,'proposal_reason':spec.proposal_reason,'proposal_templates':spec.proposal_templates,'ig_steps':spec.ig_steps},
        semantics='native_intervention_effect',nodes=tuple(nodes)+(output,),edges=tuple(edges),evidence=tuple(evidence),candidates=tuple(candidates),
        approximations=('Predeclared sites; no exhaustive circuit discovery','Independent interventions are not additive path or edge contributions','Tiny held-out controls are plumbing evidence only'),
        assumptions={'attention':'original eager attention recomputed','normalization':'original model normalization recomputed','layout':'visual arrangement has no causal meaning','dtype':'float32'},
        pruning={'rule':'predeclared explicit sites only','retained_nodes':len(nodes)+1,'omitted_node_count':None},
        omitted_contributions={'conservation_defined':False,'error_nodes':[],'reason':'Nonadditive direct effects; other sites and interactions not measured'})
    return {'graph':graph.to_dict(),'examples':examples,'scientific_verdict':'not_tested',
            'engineering_status':'rejected' if any(e.engineering_status=='rejected' for e in evidence) else 'passed'}


def run_native_route(spec):
    errors=spec.engine.planning_errors(spec)
    if errors:raise ValueError('; '.join(errors))
    with spec.engine.open_session() as session:payload=evaluate_routes(spec,session)
    return OperationExecutionResult(payload=payload,metadata={'evidence_level':'engineering','scientific_verdict':'not_tested'},
                                    example_coverage={'example_keys':[p.key for p in spec.pairs]})
