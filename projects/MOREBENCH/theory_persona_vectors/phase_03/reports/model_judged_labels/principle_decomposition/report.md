# Principle Decomposition

## 6x6 Correlation Matrix

| feature | rights_autonomy | fairness_justice | honesty_truthfulness | responsibility_accountability | loyalty_trust | virtue_character |
|---|---:|---:|---:|---:|---:|---:|
| `rights_autonomy` | 1.000 | 0.095 | 0.015 | 0.063 | -0.047 | -0.127 |
| `fairness_justice` | 0.095 | 1.000 | 0.015 | 0.003 | 0.097 | 0.235 |
| `honesty_truthfulness` | 0.015 | 0.015 | 1.000 | 0.245 | 0.071 | 0.164 |
| `responsibility_accountability` | 0.063 | 0.003 | 0.245 | 1.000 | 0.183 | 0.157 |
| `loyalty_trust` | -0.047 | 0.097 | 0.071 | 0.183 | 1.000 | 0.201 |
| `virtue_character` | -0.127 | 0.235 | 0.164 | 0.157 | 0.201 | 1.000 |

## Alpha >= 0.5 Composite Survivors

| features | k | alpha | mean inter-item r | PC2 r length-resid |
|---|---:|---:|---:|---:|

## Top PC2 Candidates (Length-Residualized)

| features | k | alpha | mean inter-item r | PC2 r | PC1 r | PC3 r |
|---|---:|---:|---:|---:|---:|---:|
| `virtue_character` | 1 | nan | nan | 0.327 | 0.165 | -0.342 |
| `rights_autonomy, loyalty_trust, virtue_character` | 3 | 0.024 | 0.009 | 0.326 | 0.101 | -0.236 |
| `loyalty_trust, virtue_character` | 2 | 0.331 | 0.201 | 0.323 | 0.135 | -0.237 |
| `rights_autonomy, responsibility_accountability, loyalty_trust, virtue_character` | 4 | 0.239 | 0.072 | 0.318 | 0.060 | -0.198 |
| `responsibility_accountability, loyalty_trust, virtue_character` | 3 | 0.395 | 0.180 | 0.309 | 0.082 | -0.193 |
| `rights_autonomy, fairness_justice, responsibility_accountability, loyalty_trust, virtue_character` | 5 | 0.318 | 0.086 | 0.308 | 0.005 | -0.199 |
| `fairness_justice, responsibility_accountability, loyalty_trust, virtue_character` | 4 | 0.394 | 0.146 | 0.301 | 0.021 | -0.195 |
| `rights_autonomy, honesty_truthfulness, loyalty_trust, virtue_character` | 4 | 0.160 | 0.046 | 0.300 | 0.030 | -0.191 |
| `rights_autonomy, fairness_justice, loyalty_trust, virtue_character` | 4 | 0.249 | 0.076 | 0.298 | 0.025 | -0.218 |
| `rights_autonomy, fairness_justice, honesty_truthfulness, responsibility_accountability, loyalty_trust, virtue_character` | 6 | 0.373 | 0.091 | 0.296 | -0.032 | -0.174 |
| `rights_autonomy, honesty_truthfulness, responsibility_accountability, loyalty_trust, virtue_character` | 5 | 0.341 | 0.093 | 0.294 | 0.008 | -0.165 |
| `fairness_justice, loyalty_trust, virtue_character` | 3 | 0.382 | 0.178 | 0.293 | 0.045 | -0.217 |
| `rights_autonomy, fairness_justice, honesty_truthfulness, loyalty_trust, virtue_character` | 5 | 0.275 | 0.072 | 0.292 | -0.020 | -0.191 |
| `honesty_truthfulness, loyalty_trust, virtue_character` | 3 | 0.327 | 0.145 | 0.290 | 0.050 | -0.185 |
| `fairness_justice, honesty_truthfulness, responsibility_accountability, loyalty_trust, virtue_character` | 5 | 0.432 | 0.137 | 0.288 | -0.020 | -0.169 |
| `responsibility_accountability, virtue_character` | 2 | 0.269 | 0.157 | 0.287 | 0.073 | -0.228 |
| `rights_autonomy, virtue_character` | 2 | -0.288 | -0.127 | 0.285 | 0.093 | -0.281 |
| `fairness_justice, honesty_truthfulness, loyalty_trust, virtue_character` | 4 | 0.359 | 0.131 | 0.284 | -0.006 | -0.187 |
| `honesty_truthfulness, responsibility_accountability, loyalty_trust, virtue_character` | 4 | 0.448 | 0.170 | 0.283 | 0.023 | -0.158 |
| `rights_autonomy, responsibility_accountability, virtue_character` | 3 | 0.095 | 0.031 | 0.282 | 0.042 | -0.219 |
