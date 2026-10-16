"""Bounded native-neuron summaries and evidence-linked manual dossiers."""
from .summaries import QUANTITIES, SummaryLimits, summarize_observations
from .dossiers import ExplanationCandidate, ExplanationProvider, StoredExplanationProvider, ManualExplanationProvider, build_dossiers
