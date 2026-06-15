"""Defines pipeline for splitting datasets."""

from pathlib import Path
from typing import Dict

from flux_pipelines.sdk.component.prebuilt import PrebuiltComponent
from kfp import dsl, kubernetes


@dsl.pipeline(
    name="splitting",
    description="Pipeline that loads preprocessed data and splits it into train, validation, and test sets.",
)
def pipeline(splitting_config: Dict):
    """Loads preprocessed data and splits it into train, validation, and test sets."""
    base = Path(__file__).parent.parent / "components"
    splitting_component = PrebuiltComponent.from_pyproject_toml(
        base, "splitting", input_artifacts=[]
    )

    splitting_task = splitting_component(config=splitting_config)

    kubernetes.add_pod_label(splitting_task, "access-ml-pipeline", "true")
    splitting_task.set_memory_request("16Gi")
    splitting_task.set_memory_limit("32Gi")
    splitting_task.set_cpu_request("4")
    splitting_task.set_cpu_limit("8")
