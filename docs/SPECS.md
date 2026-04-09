# Specs

Workflow specs are canonical in Neon via `workflow_specs`, with checked-in
snapshots stored per phase under `projects/<project>/phases/<phase>/specs/`.

## Shape

```json
{
  "id": "demo_spec",
  "name": "Demo Spec",
  "description": "Short description",
  "version": 1,
  "dataset": {
    "source": {
      "mode": "table",
      "table": "interp_examples_v0"
    },
    "filters": {
      "sql_where": "label_quality IN ('high', 'medium')"
    },
    "label": {
      "mode": "direct",
      "expression_sql": "decision_type"
    },
    "split": {
      "mode": "random_stratified",
      "train_pct": 70,
      "val_pct": 15,
      "test_pct": 15
    },
    "probe_defaults": {
      "data_source": "router",
      "pooling": "last_token",
      "n_folds": 5
    },
    "identity": {
      "column": "log_id"
    },
    "publish_mode": "view"
  },
  "capture": {
    "model": "Qwen/Qwen3-8B",
    "layers": [16, 24, 32],
    "pooling": "mean_pool",
    "router": true,
    "residual": true
  },
  "analysis": {
    "methods": ["probe"],
    "targets": ["workflow_label"],
    "data_source": "router",
    "pooling": "last_token",
    "group_column": "matched_pair_id"
  },
  "report": {
    "output_dir": "projects/<project>/phases/<phase>/reports"
  }
}
```

## Source Modes

- `table`
  - use a named Neon relation
- `sql`
  - use a read-only `SELECT`/`WITH` query

## Label Modes

- `direct`
  - label is the value of `expression_sql`
- `binary_rule`
  - `expression_sql` is treated as a boolean rule and mapped to two classes
- `bucket`
  - `expression_sql` is bucketed into named ranges

## Notes

- Dataset build publishes a named Neon relation.
- Workflow publications should expose a stable `workflow_row_key`; set
  `dataset.identity.column` when `log_id` is synthetic or otherwise not durable.
- Workflow publications also expose `workflow_prompt_hash` so capture reuse can
  detect prompt drift.
- Workflow publications may also expose grouping columns such as
  `matched_pair_id`. Analysis can export these columns and use them for grouped
  evaluation.
- Capture reads from that published relation.
- Analysis can target `workflow_label` directly.
- Analysis can also use `analysis.group_column` or `--group-column` to keep
  dependent rows in the same fold during grouped CV. This should be preferred
  over random row-level splits when rows share prompt structure.
- Legacy `prep-targets` are deprecated and are no longer part of the canonical model.
