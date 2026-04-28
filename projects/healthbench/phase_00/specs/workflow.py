"""Phase 00 placeholder for HealthBench benchmark framing.

Phase 00 has no executable capture or analysis workflow yet. The first
executable workflow should be introduced after Phase 01 freezes the latent-label
ontology and the HealthBench Consensus metadata table exists in Neon.
"""

from __future__ import annotations

MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
PROJECT = "healthbench"
PHASE = "phase_00"
INITIAL_SLICE = "healthbench_consensus"
PLANNED_CONSENSUS_TABLE = "healthbench_consensus_v1"


def build_workflow():
    raise NotImplementedError(
        "HealthBench Phase 00 is a validation/framing phase. "
        "Create the first executable workflow after Phase 01 label freeze."
    )
