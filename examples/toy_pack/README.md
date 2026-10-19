# Toy experiment pack

This fixture exercises infrastructure only. It runs two deterministic ToyEngine
examples locally, maps two labels and builds a report. Reference and scale are
disabled; setup and smoke share the same tiny workflow. No Modal or GPU calls.

From the repository root, using your installed Xenon Python environment:

```bash
python -m pipelines_v2.cli workflow plan --file examples/toy_pack/workflow.py
python -m pipelines_v2.cli workflow run --file examples/toy_pack/workflow.py
python -m pipelines_v2.cli workflow runs --file examples/toy_pack/workflow.py
python -m pipelines_v2.cli workflow show --run-id <run-id> --local-catalog-root examples/toy_pack/.catalog
python -m pipelines_v2.cli workflow report <run-id> --local-catalog-root examples/toy_pack/.catalog
python examples/toy_pack/report_evidence.py <run-id>
```

The last command reads the actual completed label-map artifact and records the
refutation of the deliberately false fixture assertion “three rows were mapped”.
It demonstrates that refuted science and successful engineering are separate.
Its verdict is engineering-only and supports no model or paper claim. The
catalog records the report artifact and its direct source input. The original
workflow source and fixture notes stay unchanged by reporting.

Keep `pack.json` source/model/data/preset hashes synchronized deliberately when
changing the experiment. `workflow_locks(unchecked_workflow())` produces the
model/data recipe locks; `unchecked_workflow().semantic_hash()` produces the
preset lock. Never regenerate locks silently as part of normal execution.

The fixture uses a workspace FileCatalog under `.catalog/` plus the CLI's
ordinary local mirror. Real studies keep the shared team catalog defaults.
Generated artifact/catalog/report directories are ignored. Clone this structure,
not its fixture results, for a study. Use the study’s chosen record format.
