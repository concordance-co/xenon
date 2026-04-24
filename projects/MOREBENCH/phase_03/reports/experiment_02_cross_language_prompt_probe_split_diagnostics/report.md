# Experiment 02 Cross-Language Prompt Probe Split Diagnostics

- capture artifact: `capture_1_0c975385595d`

## Calibration Summary
- `virtue_ethics__vs__generic_ethics_control`: probe AUROC `1.0`, probe bal acc @0.5 `0.5139`, probe bal acc @train-opt `0.5139`, probe bal acc @test-opt `1.0`, text bal acc @0.5 `0.6528`, oracle rescue `0.4861`
- `deontology__vs__contractualism`: probe AUROC `1.0`, probe bal acc @0.5 `0.5806`, probe bal acc @train-opt `0.5806`, probe bal acc @test-opt `1.0`, text bal acc @0.5 `0.5`, oracle rescue `0.4194`
- `contractarianism__vs__generic_ethics_control`: probe AUROC `1.0`, probe bal acc @0.5 `0.5806`, probe bal acc @train-opt `0.5806`, probe bal acc @test-opt `1.0`, text bal acc @0.5 `0.5583`, oracle rescue `0.4194`
- `utilitarian__vs__generic_ethics_control`: probe AUROC `1.0`, probe bal acc @0.5 `0.75`, probe bal acc @train-opt `0.75`, probe bal acc @test-opt `1.0`, text bal acc @0.5 `0.6111`, oracle rescue `0.25`
- `contractualism__vs__generic_ethics_control`: probe AUROC `1.0`, probe bal acc @0.5 `0.7555`, probe bal acc @train-opt `0.7555`, probe bal acc @test-opt `1.0`, text bal acc @0.5 `0.6444`, oracle rescue `0.2445`
- `virtue_ethics__vs__contractarianism`: probe AUROC `1.0`, probe bal acc @0.5 `0.7639`, probe bal acc @train-opt `0.7639`, probe bal acc @test-opt `1.0`, text bal acc @0.5 `0.5583`, oracle rescue `0.2361`
- `utilitarian__vs__contractualism`: probe AUROC `1.0`, probe bal acc @0.5 `0.7694`, probe bal acc @train-opt `0.7694`, probe bal acc @test-opt `1.0`, text bal acc @0.5 `0.6555`, oracle rescue `0.2306`
- `contractarianism__vs__contractualism`: probe AUROC `1.0`, probe bal acc @0.5 `0.7778`, probe bal acc @train-opt `0.7778`, probe bal acc @test-opt `1.0`, text bal acc @0.5 `0.5639`, oracle rescue `0.2222`
- `utilitarian__vs__contractarianism`: probe AUROC `1.0`, probe bal acc @0.5 `0.7861`, probe bal acc @train-opt `0.7861`, probe bal acc @test-opt `1.0`, text bal acc @0.5 `0.5722`, oracle rescue `0.2139`
- `deontology__vs__contractarianism`: probe AUROC `1.0`, probe bal acc @0.5 `0.7917`, probe bal acc @train-opt `0.7917`, probe bal acc @test-opt `1.0`, text bal acc @0.5 `0.575`, oracle rescue `0.2083`
- `deontology__vs__utilitarian`: probe AUROC `1.0`, probe bal acc @0.5 `0.8056`, probe bal acc @train-opt `0.8056`, probe bal acc @test-opt `1.0`, text bal acc @0.5 `0.5861`, oracle rescue `0.1944`
- `virtue_ethics__vs__utilitarian`: probe AUROC `1.0`, probe bal acc @0.5 `0.8333`, probe bal acc @train-opt `0.8333`, probe bal acc @test-opt `1.0`, text bal acc @0.5 `0.6278`, oracle rescue `0.1667`
- `deontology__vs__generic_ethics_control`: probe AUROC `0.9998`, probe bal acc @0.5 `0.8583`, probe bal acc @train-opt `0.8583`, probe bal acc @test-opt `0.9972`, text bal acc @0.5 `0.5861`, oracle rescue `0.1389`
- `virtue_ethics__vs__contractualism`: probe AUROC `1.0`, probe bal acc @0.5 `0.8639`, probe bal acc @train-opt `0.8639`, probe bal acc @test-opt `1.0`, text bal acc @0.5 `0.6639`, oracle rescue `0.1361`
- `deontology__vs__virtue_ethics`: probe AUROC `1.0`, probe bal acc @0.5 `0.925`, probe bal acc @train-opt `0.925`, probe bal acc @test-opt `1.0`, text bal acc @0.5 `0.6111`, oracle rescue `0.075`

## Source-Family + Cross-Language Holdout Summary
- `deontology__vs__virtue_ethics`: text AUROC `0.6244`, text bal acc `0.6472`, probe AUROC `1.0`, probe bal acc `0.925`, delta AUROC `0.3756`, delta bal acc `0.2778`
- `virtue_ethics__vs__contractarianism`: text AUROC `0.6428`, text bal acc `0.5556`, probe AUROC `1.0`, probe bal acc `0.7667`, delta AUROC `0.3572`, delta bal acc `0.2111`
- `deontology__vs__generic_ethics_control`: text AUROC `0.62`, text bal acc `0.5944`, probe AUROC `0.9994`, probe bal acc `0.8`, delta AUROC `0.3794`, delta bal acc `0.2056`
- `virtue_ethics__vs__contractualism`: text AUROC `0.6233`, text bal acc `0.6639`, probe AUROC `1.0`, probe bal acc `0.8528`, delta AUROC `0.3767`, delta bal acc `0.1889`
- `virtue_ethics__vs__utilitarian`: text AUROC `0.6306`, text bal acc `0.6417`, probe AUROC `1.0`, probe bal acc `0.8306`, delta AUROC `0.3694`, delta bal acc `0.1889`
- `contractarianism__vs__contractualism`: text AUROC `0.6847`, text bal acc `0.5639`, probe AUROC `1.0`, probe bal acc `0.7528`, delta AUROC `0.3153`, delta bal acc `0.1889`
- `utilitarian__vs__contractarianism`: text AUROC `0.6733`, text bal acc `0.5444`, probe AUROC `1.0`, probe bal acc `0.7278`, delta AUROC `0.3267`, delta bal acc `0.1834`
- `deontology__vs__contractarianism`: text AUROC `0.6378`, text bal acc `0.5889`, probe AUROC `1.0`, probe bal acc `0.7556`, delta AUROC `0.3622`, delta bal acc `0.1667`
- `deontology__vs__utilitarian`: text AUROC `0.6744`, text bal acc `0.625`, probe AUROC `1.0`, probe bal acc `0.7861`, delta AUROC `0.3256`, delta bal acc `0.1611`
- `utilitarian__vs__contractualism`: text AUROC `0.6761`, text bal acc `0.6583`, probe AUROC `1.0`, probe bal acc `0.7722`, delta AUROC `0.3239`, delta bal acc `0.1139`
- `utilitarian__vs__generic_ethics_control`: text AUROC `0.6283`, text bal acc `0.6194`, probe AUROC `1.0`, probe bal acc `0.7306`, delta AUROC `0.3717`, delta bal acc `0.1112`
- `contractualism__vs__generic_ethics_control`: text AUROC `0.6172`, text bal acc `0.6472`, probe AUROC `1.0`, probe bal acc `0.7389`, delta AUROC `0.3828`, delta bal acc `0.0917`
- `deontology__vs__contractualism`: text AUROC `0.655`, text bal acc `0.5306`, probe AUROC `1.0`, probe bal acc `0.5861`, delta AUROC `0.345`, delta bal acc `0.0555`
- `contractarianism__vs__generic_ethics_control`: text AUROC `0.6322`, text bal acc `0.55`, probe AUROC `1.0`, probe bal acc `0.5833`, delta AUROC `0.3678`, delta bal acc `0.0333`
- `virtue_ethics__vs__generic_ethics_control`: text AUROC `0.6717`, text bal acc `0.6528`, probe AUROC `1.0`, probe bal acc `0.5222`, delta AUROC `0.3283`, delta bal acc `-0.1306`
