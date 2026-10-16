"""Set XENON_LLM_CONFIG to a JSON configuration; use the ordinary workflow CLI."""
import json
import os
from pathlib import Path
from pipelines_v2.research import llm
from pipelines_v2.api import Dataset

def settings():return json.loads(Path(os.environ['XENON_LLM_CONFIG']).read_text())
def build_dataset():return Dataset.from_examples(())
def build_workflow(dataset=None):return llm.build_workflow(settings())
def build_runner_specs():return llm.runner_specs(settings())
