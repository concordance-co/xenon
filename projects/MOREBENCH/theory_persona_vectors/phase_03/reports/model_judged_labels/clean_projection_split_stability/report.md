# Clean Projection Split Stability

- trials: `100`
- split: 15 dilemmas train / 15 dilemmas test
- target: procedural_minus_decisive, length-residualized
- basis: L32 first16, all18, positive-vs-neutral-only projected out

| metric | mean | median | p05 | p95 |
|---|---:|---:|---:|---:|
| `train_pc1` | 0.020 | 0.006 | -0.314 | 0.299 |
| `train_pc2` | -0.000 | -0.004 | -0.251 | 0.251 |
| `train_pc3` | -0.002 | 0.018 | -0.296 | 0.291 |
| `test_pc1` | 0.003 | 0.093 | -0.277 | 0.267 |
| `test_pc2` | 0.002 | 0.013 | -0.280 | 0.256 |
| `test_pc3` | -0.004 | 0.006 | -0.310 | 0.298 |
| `train_max_abs_pc1_3` | 0.265 | 0.271 | 0.174 | 0.340 |
| `test_max_abs_pc1_3` | 0.256 | 0.252 | 0.184 | 0.335 |
