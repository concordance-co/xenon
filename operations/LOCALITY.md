# Data and storage

Choose storage with the execution mode. The rules below describe team practice
and the limits that code enforces. See [execution modes](README.md#select-compute-and-storage).

## Storage by mode

| Data | Shared Modal workflow | Ordinary local workflow | Offline scratch |
| --- | --- | --- | --- |
| Dataset | Neon/Postgres is the team default. | Explicit local inputs or configured data source. | Local fixtures only; no deferred or remote data. |
| Activations and generations | Modal artifact volume. | Configured `LocalArtifactStore`. | Private local attempt, then verified local publication. |
| Model and tool assets | Prepared worker image or mounted cache. | Prepared assets at the configured paths. | Prepared offline cache; no downloads. |
| Runs and artifact manifests | Configured catalog plus CLI local mirror. | Configured catalog plus CLI local mirror. | Local catalog only. |
| Reports | Built locally from selected results. | Built locally. | Built within the scratch limits. |

The default local catalog is `$XENON_HOME/pipelines_v2/catalog`.
`XENON_HOME` defaults to `~/.xenon`. Ordinary local artifact paths come from the runner.
Scratch artifacts default to `$XENON_HOME/pipelines_v2/artifacts`.

## Shared work

Use Neon for durable team datasets. Record the source table, version, and selection.
Use repeatable import code when a dataset comes from an external source.
Use a new version when changed data must remain reproducible.

Use Modal volumes for shared captures and analysis outputs.
The current recipes use `xenon-data` for artifacts and `xenon-models` for model caches.
Volume names and mount paths are configuration, not universal constants.
Keep analysis near bulk tensors. Pass typed artifact refs between steps.

Use the catalog to find runs and artifacts. A file on a volume does not prove
that the artifact is complete or indexed. See [discovery](INDEXING.md).

## Local work and transfers

Local capture and analysis are supported. They do not require a scratch profile.
Ordinary local execution does not provide scratch isolation or its resource limits.
Select the mode that meets the task's compute, storage, and network requirements.

Read the fields needed for the task through typed artifact handles or the evidence reader.
Avoid downloading a full remote capture to obtain a small statistic or plot.
Compute the reduction on Modal, then read the result locally.
Keep exports and report inputs within an explicit size budget.

`ModalVolumeStore.transfer_policy` rejects transfers with an unknown size or a size
above its configured limit, unless an override is set. The code checks size;
it does not decide whether a transfer serves the research task.
Set an override only for an intentional transfer with known scope and available storage.

Shared images and latent tensors use the same volume and transfer model.
Reports can use bounded thumbnails or summaries. See [image contracts](../platform/IMAGE_DATA.md).

## Code-enforced limits

- [Scratch](../platform/LOCAL_SCRATCH.md) enforces CPU execution, offline access, supported inputs, and explicit resource limits.
- [Transfer policy](../pipelines_v2/storage/modal.py) checks remote read sizes.
- [Completion checks](../platform/DURABILITY.md) verify stored payloads before eligible reuse.
- [Publication](../platform/PUBLICATION.md) checks supported artifact types, dependencies, and transfer budgets.

These limits are distinct from the team's choice to keep shared datasets in Neon.
Do not assume that every storage convention is enforced by a constructor.

## Repository contents

Keep source, configuration, small example fixtures, and report templates in Xenon.
Keep generated artifact stores, captures, checkpoints, model weights, and credentials out of commits.
Use the consuming research repo for study-specific records and chosen deliverables.

Publication is an explicit operation. Local artifacts do not become shared automatically.
Use [the publication procedure](../platform/PUBLICATION.md) for supported artifacts.
