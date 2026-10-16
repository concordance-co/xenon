# Evidence workbench

Browse existing catalog records and bounded result summaries without launching a
model. The public query module imports without Torch, Diffusers, Modal or UI
packages. Modal is loaded only when explicitly reading a remote summary.

```bash
python -m pipelines_v2.evidence runs --study native --limit 10
python -m pipelines_v2.evidence show wr_example
python -m pipelines_v2.evidence artifact transform_example --result
python -m pipelines_v2.evidence report transform_example attribution_example \
  --run-id wr_example --output /tmp/evidence.html
python -m pipelines_v2.evidence compare report_first report_second --metric margin
```

The same commands are available under `python -m pipelines_v2.cli evidence`.
Place `--catalog-root /path/to/existing/catalog` before the subcommand to browse
a local-scratch run. The default is the normal local catalog mirror; this reader
does not create a separate index or silently contact the shared database.

```python
from pipelines_v2.evidence import EvidenceReader, ViewLimits, compare_evidence
from pipelines_v2.evidence.catalog import local_catalog
from pipelines_v2.evidence.report import write_workbench

reader = EvidenceReader(local_catalog(), limits=ViewLimits())
page = reader.runs(study="steering", status="completed", limit=25)
summary = reader.artifact("transform_example", include_result=True)
write_workbench("/tmp/evidence.html", [summary])
```

Any existing catalog implementation can be passed to `EvidenceReader`. Run
discovery accepts study/workflow-name substring, model substring, exact operation
kind and status. Pagination is explicit. The existing catalog API determines
metadata scan cost for custom catalogs. The default read-only FileCatalog wrapper
reads at most 10,001 run files and 16 MiB of metadata (1 MiB per file), before
applying filters, and marks truncation. Results are sorted within that discovery
window; a truncated scan is not a globally complete newest-first listing. It
creates no index or directories. Result payload transfer is bounded separately.

The report has neuron context/explanation/intervention panels, typed graph edges
and controls, paired image previews and provenance details. A run view includes
failures and the stored workflow configuration. It does not invent a runnable
source path: replay needs the original `workflow.py` and its pinned dependencies.
Missing files, inaccessible volumes, unknown IDs and oversized summaries fail
with an explicit diagnostic. Opening an HTML report has no network or execution
side effects.

Default view bounds are 16 MiB of actual JSON result bytes across the reader,
100 examples, 100 graph nodes/300 edges and 32 images. Raw capture and generation
sets are rejected. Graphs or contexts exceeding the caps require an explicitly
reduced upstream summary. Image previews must already be embedded JPEGs; the
viewer never fetches raw images or activation tensors. Summaries with C03 hashes
are verified as read; a verified summary is not verification of every upstream
payload or hardware qualification.

Comparison requires exactly one explicit C05 record for the requested metric
per artifact and complete execution identity. New ReportSpec evidence payloads
preserve the full pack under `result.evidence`; standalone bundles are also
recognized. The reader validates ExperimentPack and every EvidenceRecord, then
checks each linked catalog manifest's run, workflow preset hash, step/kind and
resolved provenance. Result selectors read only declared JSON, with checksum
verification, a 256 KiB individual and 1 MiB combined cap; at most 100 direct
artifacts are consulted, without recursive evidence validation. These reads also
count toward the reader-wide 16 MiB budget. Missing packs, incomplete identities,
bad digests, inaccessible sources or mismatched selectors remain visible as
unverified and cannot compare. Evaluator, split, pack, preset/question, evidence
level, metric units, model/tokenizer, methods, numerical and data semantics must match. A
difference is displayed as non-comparable; there is no automatic averaging,
cross-model equivalence policy, inferred metric or scientific promotion.

Actual integration evidence includes Qwen3 neuron dossiers, a native
route graph and an SDXL steering gallery. Full scientific study quality is
outside this plumbing check; interpretation of displayed evidence is
study-specific.

C05 link verification is one hop only. `compare_evidence` consumes the trusted
in-memory views returned by `EvidenceReader`; its verification marker is not an
authentication format for arbitrary caller-authored or deserialized Python dicts.
Changing a verified bundle invalidates its recorded digest. Nested evidence
records obey the same caller-selected example cap as standalone records.
