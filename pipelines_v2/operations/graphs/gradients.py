"""Candidate-only embeddings: true autograd ancestors, never selected head captures."""

def embedding_candidates(scalar, embeddings, *, steps=8):
    import torch
    if type(steps)is not int or steps<1:raise ValueError('IG grid must contain positive intervals')
    if not torch.isfinite(embeddings).all():raise ValueError('Nonfinite embedding input')
    x=embeddings.detach().clone().requires_grad_(True)
    output=scalar(x)
    gradient=torch.autograd.grad(output,x)[0]
    baseline=torch.zeros_like(x)
    total=torch.zeros_like(x)
    for i in range(steps+1):
        point=(baseline+(x.detach()-baseline)*(i/steps)).requires_grad_(True)
        grad=torch.autograd.grad(scalar(point),point)[0]
        total+=grad.detach()*(.5 if i in (0,steps) else 1.)
    ig=(x.detach()-baseline)*total/steps
    with torch.no_grad():delta=float(scalar(x.detach())-scalar(baseline))
    if not torch.isfinite(ig).all() or not torch.isfinite(gradient).all():raise ValueError('Nonfinite candidate gradients')
    return {'semantics':'candidate_ranking', 'basis':'input_token_embedding', 'baseline':'zero token embeddings; positions and mask unchanged',
        'integration':'trapezoidal', 'grid':[i/steps for i in range(steps+1)],
        'input_times_gradient':(x.detach()*gradient.detach()).sum(-1).squeeze(0).tolist(),
        'integrated_gradients':ig.sum(-1).squeeze(0).tolist(), 'target_delta':delta,
        'completeness_residual':delta-float(ig.sum()), 'scientific_verdict':'not_tested'}
