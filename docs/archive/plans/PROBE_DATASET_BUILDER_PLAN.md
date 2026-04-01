## Probe Dataset Builder + Controlled Variables + Versioned Probe Runs

### Summary
Build a 2-step workflow in the dashboard UI:
1. **Prepare Probe Dataset** (labeling, controlled-variable filters, sample size, train/val/test split, preview, versioned materialization).
2. **Run Probe** (select prepared dataset artifact + layers + pooling + source, then run analysis).

This will make “risk 1 vs 5 for only record observations / only buys / only sells / only trades” first-class, reproducible, and non-destructive.

### Implementation Changes

1. **Backend: dataset preview/materialization APIs (read-only SQL + file outputs only)**
- Add endpoints in `pipelines/backend/app.py`:
  - `POST /probe-datasets/preview`
  - `POST /probe-datasets/materialize`
  - `GET /probe-datasets`
  - `GET /probe-datasets/{id}`
- `preview` returns:
  - available counts after filters
  - class distribution
  - proposed split counts
  - sample rows for inspection
  - warning when requested N > available (auto-uses max available)
- `materialize` writes versioned artifacts to:
  - `/data/interp_exports/probe_runs/<run_id>/dataset.parquet`
  - `/data/interp_exports/probe_runs/<run_id>/manifest.json`
- Keep DB strictly read-only: all transformations/splitting happen query-time/in-memory; only artifact files are written.

2. **Controlled variables support (required)**
- Add a **controlled-variable filter block** in dataset spec:
  - `decision_scope` presets:
    - `all`
    - `record_observation_only`
    - `trade_only`
    - `buy_only`
    - `sell_only`
  - optional additional filters:
    - `vault_risk_preference` set/range (for risk 1 vs 5)
    - `date range` on `created_at`
    - optional `sql_where` (advanced mode)
- Enforce filters into one validated read-only query pipeline before labeling/splitting.
- Include applied controlled variables in manifest for reproducibility.

3. **Labeling + sample budget**
- Add label builder options:
  - built-in templates (`decision_type`, `trade_side`, `risk_1_vs_5`, etc.)
  - custom SQL expression mode (advanced)
- Materialization behavior:
  - starts from high-quality eligible rows
  - applies controlled-variable filters
  - applies label validity filter
  - applies requested example budget (`target_examples`) with fallback to max available
  - optional class balancing (`none` or `stratified_per_label`)

4. **Explicit train/val/test split artifacts**
- Add split config:
  - `train_pct`, `val_pct`, `test_pct`, `seed`
  - stratify by label (default on)
  - optional group holdout key (`vault_address`) to reduce leakage
- Write split assignment column into artifact (`split` = train/val/test).
- Manifest stores row counts and label distributions per split.

5. **UI: new “Probe Dataset” workflow + run handoff**
- Extend Explorer with a dedicated **Probe Dataset** workspace:
  - label config
  - controlled variables panel (incl. decision-scope presets above)
  - example budget + split controls
  - preview table/sample
  - “Materialize Dataset” action
- Extend Analysis UI:
  - dataset artifact selector (from `/probe-datasets`)
  - existing layer/pooling/source controls reused
  - generated command includes selected artifact path
- Keep current default prep parquet flow intact; no overwrite of existing files.

6. **Analysis pipeline compatibility**
- Extend analysis entrypoints to accept a custom labels parquet path and split-aware evaluation mode.
- When split column exists:
  - train on train
  - report metrics on val/test
  - persist split metrics in result outputs
- Maintain backward compatibility with existing CV-only path when no split is provided.

### Public Interfaces / Types

1. **Probe dataset spec**
- Add `ProbeDatasetSpec` contract with:
  - `name`, `description`
  - `source` (default `interp_examples_v0` high-quality slice)
  - `label` definition (template or custom expression)
  - `controlled_variables` (decision_scope + optional filters)
  - `sampling` (`target_examples`, balancing, seed)
  - `split` (`train/val/test`, stratify, optional group key)

2. **Probe dataset artifact metadata**
- Add `ProbeDatasetArtifact`:
  - `run_id`, `name`, `created_at`
  - `dataset_path`, `manifest_path`
  - `row_count`, `class_distribution`, `split_distribution`
  - resolved controlled-variable filters and label spec

3. **No DB schema migration**
- No SQLite write-path changes.
- New persistence is file-based only under `/data/interp_exports/probe_runs`.

### Test Plan

1. **Backend API tests**
- Preview/materialize reject mutating SQL and unsafe fragments.
- `decision_scope` filter correctness:
  - record-only, trade-only, buy-only, sell-only slices validated.
- Risk 1 vs 5 example:
  - dataset contains only `vault_risk_preference IN (1,5)` and expected labels.
- Example budget fallback:
  - request > available returns available count with warning.
- Split validity:
  - disjoint train/val/test, correct percentages, stratification behavior.

2. **Artifact and reproducibility tests**
- Materialization creates versioned run folder and manifest.
- Re-running with same seed/spec yields stable split assignments (within deterministic constraints).
- Existing prep exports remain unchanged.

3. **Analysis tests**
- Split-aware run consumes artifact and reports val/test metrics.
- Backward compatibility: legacy parquet + CV path still works unchanged.

4. **UI QA scenarios**
- Build risk 1 vs 5 dataset for each decision scope preset (record, buy, sell, trade) and verify preview.
- Materialize dataset, select it in Analysis, run probe with chosen layers/pooling.
- Confirm no overwrite of previous artifacts.

### Assumptions and Defaults
1. High-quality rows are the default source slice for probe datasets.
2. Default controlled-variable scope is `all` (user can narrow to record/trade/buy/sell presets).
3. Default split is `70/15/15`, stratified by label, seed fixed unless changed.
4. Advanced custom label SQL and advanced `sql_where` are allowed but validated as read-only.
5. This phase does not add training orchestration changes beyond existing command execution patterns; it standardizes dataset creation and split-aware inputs.
