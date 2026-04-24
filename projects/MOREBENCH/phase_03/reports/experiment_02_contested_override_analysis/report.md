# Contested Override Analysis

## Coverage
- split-group rows merged across captures: `132`
- majority-defined non-generic rows used for pooled binary probes: `100`
- tie groups excluded from binary targets: `2`
- generic tracks majority in `16/20` majority-defined groups
- generic defects in `4/20` groups: `public_conflict_011`, `public_conflict_033`, `public_conflict_039`, `public_conflict_060`

## Pooled Results
### `differs_from_generic`
- text baseline AUROC: `0.4838`
- length baseline AUROC: `0.5628`
- prime-only baseline AUROC: `0.4191`
- best probe layer: `4`
- best probe AUROC: `0.6385`
- probe minus text AUROC: `0.1547`
- probe minus prime-only AUROC: `0.2194`

### `defect_from_majority`
- text baseline AUROC: `0.481`
- length baseline AUROC: `0.4906`
- prime-only baseline AUROC: `0.5205`
- best probe layer: `4`
- best probe AUROC: `0.5708`
- probe minus text AUROC: `0.0898`
- probe minus prime-only AUROC: `0.0503`

## Within-Prime Results
### `differs_from_generic`
- `deontology`: text AUROC `0.4141`, length AUROC `0.6667`, best probe layer `32` AUROC `0.7172`, delta `0.3031`
- `utilitarian`: text AUROC `0.2976`, length AUROC `0.0595`, best probe layer `40` AUROC `0.5595`, delta `0.2619`

### `defect_from_majority`
- `deontology`: text AUROC `0.4848`, length AUROC `0.2323`, best probe layer `4` AUROC `0.5455`, delta `0.0607`
- `utilitarian`: text AUROC `0.5`, length AUROC `0.3021`, best probe layer `0` AUROC `0.6667`, delta `0.1667`

## PCA Batch Check
- layer `0`: batch silhouette `0.0004`, prime silhouette `0.0099`, override-status silhouette `-0.0048`
- layer `8`: batch silhouette `0.002`, prime silhouette `0.0428`, override-status silhouette `-0.0009`
- layer `44`: batch silhouette `0.009`, prime silhouette `0.0201`, override-status silhouette `-0.0084`
