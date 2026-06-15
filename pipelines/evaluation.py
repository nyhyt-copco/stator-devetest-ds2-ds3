"""Defines pipeline for model evaluation."""

from pathlib import Path
from typing import Dict

from flux_pipelines.sdk.component.prebuilt import PrebuiltComponent
from kfp import dsl, kubernetes


@dsl.pipeline(
    name="evaluation",
    description="Pipeline that loads prediction datasets and evaluates them.",
)
def pipeline(evaluation_config: Dict):
    """Loads prediction datasets and evaluates them."""
    base = Path(__file__).parent.parent / "components"
    evaluation_component = PrebuiltComponent.from_pyproject_toml(
        base, "evaluation", input_artifacts=[]
    )

    evaluation_task = evaluation_component(config=evaluation_config)
    kubernetes.add_pod_label(evaluation_task, "access-ml-pipeline", "true")
    evaluation_task.set_memory_request("16Gi")
    evaluation_task.set_memory_limit("32Gi")
    evaluation_task.set_cpu_request("8")
    evaluation_task.set_cpu_limit("16")
    evaluation_task.set_env_variable("PXD_EXECUTOR_BACKEND", "ray")
    evaluation_task.set_env_variable("PXD_N_CPUS", "8")
