# Scripts

The `scripts/` tree is organized by job type rather than by phase number.

## Layout

- `scripts/db/`
  - Neon helpers and one-off DB migrations
- `scripts/manifests/`
  - capture manifest builders and reconciliation helpers
- `scripts/research/`
  - real-data research payload builders
- `scripts/datasets/synthetic_market/`
  - synthetic market dataset builders and per-phase prep entrypoints
- `scripts/reports/actionability/`
  - actionability report chart generators
- `scripts/reports/decision_structure/`
  - decision-structure report chart generators
- `scripts/reports/research/`
  - real-DX / postmarket / research report chart generators
- `scripts/reports/synthetic/`
  - early synthetic phase report chart generators
- `scripts/reports/synthetic_market/`
  - synthetic market phase report chart generators

## Kept At Top Level

- `scripts/modal_capture.sh`
- `scripts/modal_restore_db.sh`
- `scripts/xenon_backend.sh`

Those stay at the top because they are operator-facing entrypoints used directly in docs, the dashboard, and shell workflows.
