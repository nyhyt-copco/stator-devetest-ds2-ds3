"""Defines pipeline for model training."""

from pathlib import Path
from typing import Dict

from flux_pipelines.sdk.component.prebuilt import PrebuiltComponent
from kfp import dsl, kubernetes


@dsl.pipeline(
    name="modelling",
    description="Pipeline that loads split datasets, trains a model, and outputs predictions.",
)
def pipeline(modelling_config: Dict):
    """Loads split datasets, trains a model, and outputs model and predictions."""
    base = Path(__file__).parent.parent / "components"
    modelling_component = PrebuiltComponent.from_pyproject_toml(
        base, "modelling", input_artifacts=[]
    )

    modelling_task = modelling_component(config=modelling_config)
    kubernetes.add_pod_label(modelling_task, "access-ml-pipeline", "true")
    modelling_task.set_memory_request("32Gi")
    modelling_task.set_memory_limit("64Gi")
    modelling_task.set_cpu_request("4")
    modelling_task.set_cpu_limit("8")
    modelling_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")
