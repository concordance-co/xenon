# Dilemma Family Keyword Variant Preflight

Phase-02 shortcut-stress-test preflight for four dilemma-family labels.
Variants are applied to dilemma text only. Char-TFIDF transfer is the exit gate.

## Exit Gate Outcomes

- `close_relationship_obligation`: **iterate_variant_design**
- `privacy_monitoring_conflict`: **iterate_variant_design**
- `disclosure_transparency_conflict`: **iterate_variant_design**
- `institutional_policy_constraint`: **iterate_variant_design**

## Per-Label Results

### `close_relationship_obligation`

construct: dilemma tension involves kinship or intimate social ties

- True counts: variant_a=190, variant_b=17, variant_c=5 (union=193, intersection=0)

**Within-variant CV (char-TFIDF on dilemma text):**

| variant | status | pos | neg | BA | AUROC |
|---|---|---:|---:|---:|---:|
| variant_a | ok | 190 | 310 | 0.843 | 0.908 |
| variant_b | ok | 17 | 483 | 0.526 | 0.824 |
| variant_c | insufficient_support | 5 | 495 |  |  |

**Cross-variant char-TFIDF transfer (train on src label, predict dst label):**

| src | dst | status | src_pos | dst_pos | BA (dst) | AUROC (dst) | agreement |
|---|---|---|---:|---:|---:|---:|---:|
| variant_a | variant_b | ok | 190 | 17 | 0.713 | 0.808 | 0.642 |
| variant_a | variant_c | ok | 190 | 5 | 0.725 | 0.801 | 0.630 |
| variant_b | variant_a | ok | 17 | 190 | 0.511 | 0.785 | 0.642 |
| variant_b | variant_c | ok | 17 | 5 | 0.496 | 0.611 | 0.956 |
| variant_c | variant_a | insufficient_src_support | 5 |  |  |  |  |
| variant_c | variant_b | insufficient_src_support | 5 |  |  |  |  |

**Pairwise label agreement matrix:**

| | variant_a | variant_b | variant_c |
|---|---:|---:|---:|
| variant_a | 1.000 | 0.642 | 0.630 |
| variant_b | 0.642 | 1.000 | 0.956 |
| variant_c | 0.630 | 0.956 | 1.000 |

### `privacy_monitoring_conflict`

construct: dilemma tension involves information or observational boundary violations

- True counts: variant_a=175, variant_b=23, variant_c=3 (union=183, intersection=0)

**Within-variant CV (char-TFIDF on dilemma text):**

| variant | status | pos | neg | BA | AUROC |
|---|---|---:|---:|---:|---:|
| variant_a | ok | 175 | 325 | 0.768 | 0.864 |
| variant_b | ok | 23 | 477 | 0.606 | 0.877 |
| variant_c | insufficient_support | 3 | 497 |  |  |

**Cross-variant char-TFIDF transfer (train on src label, predict dst label):**

| src | dst | status | src_pos | dst_pos | BA (dst) | AUROC (dst) | agreement |
|---|---|---|---:|---:|---:|---:|---:|
| variant_a | variant_b | ok | 175 | 23 | 0.747 | 0.785 | 0.664 |
| variant_a | variant_c | ok | 175 | 3 | 0.490 | 0.541 | 0.656 |
| variant_b | variant_a | ok | 23 | 175 | 0.510 | 0.676 | 0.664 |
| variant_b | variant_c | ok | 23 | 3 | 0.492 | 0.267 | 0.948 |
| variant_c | variant_a | insufficient_src_support | 3 |  |  |  |  |
| variant_c | variant_b | insufficient_src_support | 3 |  |  |  |  |

**Pairwise label agreement matrix:**

| | variant_a | variant_b | variant_c |
|---|---:|---:|---:|
| variant_a | 1.000 | 0.664 | 0.656 |
| variant_b | 0.664 | 1.000 | 0.948 |
| variant_c | 0.656 | 0.948 | 1.000 |

### `disclosure_transparency_conflict`

construct: dilemma tension involves whether to reveal or withhold information

- True counts: variant_a=168, variant_b=68, variant_c=14 (union=199, intersection=2)

**Within-variant CV (char-TFIDF on dilemma text):**

| variant | status | pos | neg | BA | AUROC |
|---|---|---:|---:|---:|---:|
| variant_a | ok | 168 | 332 | 0.683 | 0.779 |
| variant_b | ok | 68 | 432 | 0.635 | 0.793 |
| variant_c | ok | 14 | 486 | 0.500 | 0.738 |

**Cross-variant char-TFIDF transfer (train on src label, predict dst label):**

| src | dst | status | src_pos | dst_pos | BA (dst) | AUROC (dst) | agreement |
|---|---|---|---:|---:|---:|---:|---:|
| variant_a | variant_b | ok | 168 | 68 | 0.670 | 0.710 | 0.688 |
| variant_a | variant_c | ok | 168 | 14 | 0.524 | 0.524 | 0.672 |
| variant_b | variant_a | ok | 68 | 168 | 0.537 | 0.682 | 0.688 |
| variant_b | variant_c | ok | 68 | 14 | 0.461 | 0.604 | 0.852 |
| variant_c | variant_a | ok | 14 | 168 | 0.500 | 0.522 | 0.672 |
| variant_c | variant_b | ok | 14 | 68 | 0.500 | 0.568 | 0.852 |

**Pairwise label agreement matrix:**

| | variant_a | variant_b | variant_c |
|---|---:|---:|---:|
| variant_a | 1.000 | 0.688 | 0.672 |
| variant_b | 0.688 | 1.000 | 0.852 |
| variant_c | 0.672 | 0.852 | 1.000 |

### `institutional_policy_constraint`

construct: dilemma tension involves formal institutional rules, policies, or organizational structures

- True counts: variant_a=231, variant_b=97, variant_c=18 (union=256, intersection=5)

**Within-variant CV (char-TFIDF on dilemma text):**

| variant | status | pos | neg | BA | AUROC |
|---|---|---:|---:|---:|---:|
| variant_a | ok | 231 | 269 | 0.783 | 0.881 |
| variant_b | ok | 97 | 403 | 0.713 | 0.852 |
| variant_c | ok | 18 | 482 | 0.497 | 0.670 |

**Cross-variant char-TFIDF transfer (train on src label, predict dst label):**

| src | dst | status | src_pos | dst_pos | BA (dst) | AUROC (dst) | agreement |
|---|---|---|---:|---:|---:|---:|---:|
| variant_a | variant_b | ok | 231 | 97 | 0.694 | 0.757 | 0.640 |
| variant_a | variant_c | ok | 231 | 18 | 0.648 | 0.691 | 0.558 |
| variant_b | variant_a | ok | 97 | 231 | 0.615 | 0.776 | 0.640 |
| variant_b | variant_c | ok | 97 | 18 | 0.660 | 0.777 | 0.798 |
| variant_c | variant_a | ok | 18 | 231 | 0.502 | 0.665 | 0.558 |
| variant_c | variant_b | ok | 18 | 97 | 0.515 | 0.712 | 0.798 |

**Pairwise label agreement matrix:**

| | variant_a | variant_b | variant_c |
|---|---:|---:|---:|
| variant_a | 1.000 | 0.640 | 0.558 |
| variant_b | 0.640 | 1.000 | 0.798 |
| variant_c | 0.558 | 0.798 | 1.000 |

## Triage Legend

- `pass`: cross-variant char-TFIDF BA <= 0.65 for every ordered pair AND pairwise agreement >= 0.70.
- `iterate_variant_design`: cross-variant BA above 0.65 but below 0.75 — variants not disjoint enough.
- `shortcut_dominated`: cross-variant BA >= 0.75 — char-TFIDF transfers; mark in known_bugs.
- `variants_incoherent`: pairwise agreement < 0.70 — variants don't measure the same construct.
- `insufficient_support`: class support too small to run transfer reliably.
