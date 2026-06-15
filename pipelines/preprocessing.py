"""Defines pipeline for preprocessing data."""

from pathlib import Path
from typing import Dict

from flux_pipelines.sdk.component.prebuilt import PrebuiltComponent
from kfp import dsl, kubernetes


@dsl.pipeline(
    name="preprocessing",
    description="Pipeline that loads samples, preprocesses them, and outputs preprocessed samples.",
)
def pipeline(preprocessing_config: Dict):
    """Loads samples, preprocesses them, and saves preprocessed samples."""
    base = Path(__file__).parent.parent / "components"
    preprocessing_component = PrebuiltComponent.from_pyproject_toml(
        base, "preprocessing", input_artifacts=[]
    )

    preprocessing_task = preprocessing_component(config=preprocessing_config)

    kubernetes.add_node_selector(preprocessing_task, "kubernetes.io/arch", "amd64")
    kubernetes.add_pod_label(preprocessing_task, "access-ml-pipeline", "true")
    preprocessing_task.set_memory_request("16Gi")
    preprocessing_task.set_memory_limit("32Gi")
    preprocessing_task.set_cpu_request("8")
    preprocessing_task.set_cpu_limit("16")
