# A real local GPT-2 experiment

Run eight frozen prompts through capture, baseline, identity, zero, matched opposite-label donor, same-label donor and seeded random donor arms. Four recipients share the same comparison coverage. Results include target log probabilities, generated continuations, exact donor keys and a local ReportSpec artifact. No Modal, remote dataset or model download occurs during execution.

The question is deliberately small: does replacing the last prompt token's block-5 residual shift Paris-versus-London preference? This is an engineering and exploratory behavioral experiment, not a paper reproduction or localized factual mechanism. See PHASE.md for observed findings and limits.

## Explicit preparation (network allowed here only)

Run all commands from the Xenon repository root. These tested commands target macOS arm64, Python 3.13.6. The strict execution sandbox requires `sandbox-exec` on macOS; Linux requires `bwrap` with unprivileged namespaces and the `core-linux` lock. Unsupported isolation fails explicitly. No MPS/CUDA qualification is claimed.

Create an isolated environment from the **complete checked-in exact runtime lock**, leaving the shared `.venv` untouched:

```bash
mkdir -p workspaces/local_research/.cache
uv venv --python 3.13.6 workspaces/local_research/.cache/runtime
python3 - <<'PY'
import json
from pathlib import Path
lock = json.loads(Path('pipelines_v2/runtime/locks/core-local.json').read_text())
Path('workspaces/local_research/.cache/requirements.txt').write_text(lock['requirements'])
PY
uv pip install --python workspaces/local_research/.cache/runtime/bin/python --no-deps -r workspaces/local_research/.cache/requirements.txt
uv pip check --python workspaces/local_research/.cache/runtime/bin/python
export XENON_LOCAL_PYTHON="$PWD/workspaces/local_research/.cache/runtime/bin/python"
export PYTHONPATH="$PWD"
```

Download the immutable checkpoint explicitly, or set `XENON_GPT2_PREPARED` to an existing copy of those exact files. The separate download's wall time depends on network/cache state; record it yourself (the original cached-asset preparation did not record download time).

```bash
export XENON_GPT2_PREPARED="$HOME/.cache/xenon/gpt2-607a30d783dfa663caf39e06633721c8d4cfcd7e"
time workspaces/local_research/.cache/runtime/bin/hf download openai-community/gpt2 \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  README.md config.json generation_config.json merges.txt model.safetensors tokenizer.json tokenizer_config.json vocab.json \
  --local-dir "$XENON_GPT2_PREPARED"
"$XENON_LOCAL_PYTHON" -m workspaces.local_research.phase_00.prepare --cache "$XENON_GPT2_PREPARED"
```

`prepare` checks all eight SHA-256 digests against model_lock.json and creates an ignored cache symlink, giving the executable workflow a portable stable recipe path. It never downloads or replaces an existing different link. Changing model bytes is rejected. Recipe hashes are separate from C01 execution-host checkpoint identity. `setup`, `reference` and `scale` presets are disabled: preparation is the explicit command above, and this pack only qualifies smoke.

## Run offline

```bash
export XENON_HOME="$HOME/.xenon-local-research"
export XENON_EXPERIMENT_PRESET=smoke
"$XENON_LOCAL_PYTHON" -m pipelines_v2.cli workflow plan --file workspaces/local_research/phase_00/workflow.py
"$XENON_LOCAL_PYTHON" -m pipelines_v2.cli workflow run \
  --file workspaces/local_research/phase_00/workflow.py --profile local-scratch \
  --scratch-model-cache workspaces/local_research/.cache/gpt2 \
  --scratch-limits-file workspaces/local_research/phase_00/limits.json \
  --run-id wr_local_research_001 > "$XENON_HOME-run.json"
```

The enforced envelope is eight unique examples, 128 input tokens, 16 generated tokens (workflow uses eight), 256 MiB new payload, 600 seconds, and 8 GiB aggregate process-tree RSS. Artifacts and FileCatalog live under XENON_HOME. Local-scratch ignores shared Postgres defaults; the explicit local runner makes planning local too. Use **a fresh run ID** for each rerun, such as `wr_local_research_002`. Whole-profile interrupted compute resume is not currently supported; do not use `workflow resume` as if it were.

## Verify and read

The following fresh process strips credential variables and verifies offline Python, subprocess Python, native network and cloud-import sentinels. It runs the same full experiment in one explicit session, checks all captures and all twelve donor replacements against independent raw Hugging Face hooks, and measures cold load separately from warm forwards. It is an engineering oracle, not an alternate production workflow.

```bash
"$XENON_LOCAL_PYTHON" -m workspaces.local_research.phase_00.verify --output "$XENON_HOME/oracle-001/result.json"
"$XENON_LOCAL_PYTHON" -m workspaces.local_research.phase_00.inspect_run \
  --run-json "$XENON_HOME-run.json" --oracle "$XENON_HOME/oracle-001/result.json" \
  --output "$XENON_HOME/REPORT.md"
"$XENON_LOCAL_PYTHON" -m pipelines_v2.cli workflow runs \
  --file workspaces/local_research/phase_00/workflow.py --local-catalog-root "$XENON_HOME/pipelines_v2/catalog"
"$XENON_LOCAL_PYTHON" -m pipelines_v2.cli workflow show \
  --run-id wr_local_research_001 --local-catalog-root "$XENON_HOME/pipelines_v2/catalog"
```

The inspector checks every C03 payload, compares deterministic generated tokens and target values against the oracle, and renders a compact reading view of the cataloged ReportSpec inputs. The canonical report and original values remain in the artifact store. Reports explicitly list incomplete C01 fields; report/summary operations are not automatically scientifically qualified.

`pack.json` freezes workflow, data, model recipe, source and evaluator hashes. Changes require deliberate regeneration of that metadata; mismatches fail before execution. Generated reports, model weights and captures are not checked in. `evidence.json` and PHASE.md point to observed integration evidence; paths from a developer machine are references, not packaged data.
