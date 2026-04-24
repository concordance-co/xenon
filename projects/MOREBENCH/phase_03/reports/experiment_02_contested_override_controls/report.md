# Contested Override Controls

## Coverage
- majority-defined non-generic rows: `100`
- generic tracks majority in `16/20` groups
- generic defector groups: `public_conflict_011`, `public_conflict_033`, `public_conflict_039`, `public_conflict_060`

## PCA Batch Check
- layer `0`: batch `0.0004`, prime `0.0099`, override `-0.0048`
- layer `8`: batch `0.002`, prime `0.0428`, override `-0.0009`
- layer `44`: batch `0.009`, prime `0.0201`, override `-0.0084`

## Pooled
### `differs_from_generic`
- full strongest baseline: `tfidf_char` AUROC `0.6431`
- full layer 8 probe AUROC: `0.6032` delta `-0.0399` CI `[-0.1452, 0.067]`
- full best raw layer: `4` AUROC `0.6385` delta `-0.0046` CI `[-0.0982, 0.0978]`
- tail strongest baseline: `tfidf_char` AUROC `0.6986`
- tail layer 8 probe AUROC: `0.6078` delta `-0.0908` CI `[-0.1883, 0.0198]`
- tail best raw layer: `40` AUROC `0.6789` delta `-0.0197` CI `[-0.1684, 0.112]`
- residualized full best: layer `4` AUROC `0.6305` delta `-0.0126`
- residualized tail best: layer `40` AUROC `0.6507` delta `-0.0479`

### `defect_from_majority`
- full strongest baseline: `prime_length` AUROC `0.5789`
- full layer 8 probe AUROC: `0.5256` delta `-0.0533` CI `[-0.1813, 0.0559]`
- full best raw layer: `4` AUROC `0.5708` delta `-0.0081` CI `[-0.1333, 0.1107]`
- tail strongest baseline: `prime_length` AUROC `0.5779`
- tail layer 8 probe AUROC: `0.4409` delta `-0.137` CI `[-0.3274, 0.0533]`
- tail best raw layer: `4` AUROC `0.5601` delta `-0.0178` CI `[-0.2029, 0.1583]`
- residualized full best: layer `4` AUROC `0.5261` delta `-0.0528`
- residualized tail best: layer `4` AUROC `0.4196` delta `-0.1583`

## Within-Prime
### `differs_from_generic / deontology`
- full strongest baseline: `tfidf_word_char_length` AUROC `0.6768`
- full layer 8 probe AUROC: `0.5758` delta `-0.101` CI `[-0.4951, 0.28]`
- full best raw layer: `32` AUROC `0.7172` delta `0.0404` CI `[-0.26, 0.3542]`
- tail strongest baseline: `length_only` AUROC `0.6667`
- tail layer 8 probe AUROC: `0.6566` delta `-0.0101` CI `[-0.375, 0.381]`
- tail best raw layer: `24` AUROC `0.8081` delta `0.1414` CI `[-0.1429, 0.4546]`
- residualized full best: layer `32` AUROC `0.697` delta `0.0202`
- residualized tail best: layer `24` AUROC `0.7576` delta `0.0909`

### `differs_from_generic / utilitarian`
- full strongest baseline: `tfidf_word` AUROC `0.2976`
- full layer 8 probe AUROC: `0.5` delta `0.2024` CI `[-0.1467, 0.5688]`
- full best raw layer: `40` AUROC `0.5595` delta `0.2619` CI `[-0.1429, 0.6313]`
- tail strongest baseline: `tfidf_char` AUROC `0.4048`
- tail layer 8 probe AUROC: `0.3452` delta `-0.0596` CI `[-0.2967, 0.1765]`
- tail best raw layer: `44` AUROC `0.4048` delta `0.0` CI `[-0.2745, 0.24]`
- residualized full best: layer `40` AUROC `0.5952` delta `0.2976`
- residualized tail best: layer `44` AUROC `0.4405` delta `0.0357`

### `defect_from_majority / deontology`
- full strongest baseline: `tfidf_char` AUROC `0.7273`
- full layer 8 probe AUROC: `0.4141` delta `-0.3132` CI `[-0.5833, -0.0707]`
- full best raw layer: `4` AUROC `0.5455` delta `-0.1818` CI `[-0.4344, 0.04]`
- tail strongest baseline: `tfidf_char` AUROC `0.3636`
- tail layer 8 probe AUROC: `0.4343` delta `0.0707` CI `[-0.202, 0.3646]`
- tail best raw layer: `44` AUROC `0.5455` delta `0.1819` CI `[-0.022, 0.404]`
- residualized full best: layer `4` AUROC `0.5152` delta `-0.2121`
- residualized tail best: layer `44` AUROC `0.5758` delta `0.2122`

### `defect_from_majority / utilitarian`
- full strongest baseline: `tfidf_word` AUROC `0.5`
- full layer 8 probe AUROC: `0.5417` delta `0.0417` CI `[-0.2088, 0.3125]`
- full best raw layer: `0` AUROC `0.6667` delta `0.1667` CI `[-0.1415, 0.4646]`
- tail strongest baseline: `tfidf_word` AUROC `0.5312`
- tail layer 8 probe AUROC: `0.4479` delta `-0.0833` CI `[-0.3214, 0.1355]`
- tail best raw layer: `8` AUROC `0.4479` delta `-0.0833` CI `[-0.3214, 0.1355]`
- residualized full best: layer `0` AUROC `0.6771` delta `0.1771`
- residualized tail best: layer `8` AUROC `0.4271` delta `-0.1041`

