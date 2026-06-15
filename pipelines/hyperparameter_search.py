"""Defines pipeline for hyperparameter search."""

from pathlib import Path
from typing import Dict

from flux_pipelines.sdk.component.prebuilt import PrebuiltComponent
from kfp import dsl, kubernetes


@dsl.pipeline(
    name="hyperparameter_search",
    description="Pipeline that runs hyperparameter search over training and validation datasets.",
)
def pipeline(hyperparameter_search_config: Dict):
    """Runs hyperparameter search and outputs results."""
    base = Path(__file__).parent.parent / "components"
    hyperparameter_search_component = PrebuiltComponent.from_pyproject_toml(
        base, "hyperparameter_search", input_artifacts=[]
    )

    hyperparameter_search_task = hyperparameter_search_component(
        config=hyperparameter_search_config
    )
    kubernetes.add_pod_label(hyperparameter_search_task, "access-ml-pipeline", "true")
    hyperparameter_search_task.set_memory_request("32Gi")
    hyperparameter_search_task.set_memory_limit("64Gi")
    hyperparameter_search_task.set_cpu_request("4")
    hyperparameter_search_task.set_cpu_limit("8")
    hyperparameter_search_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")
