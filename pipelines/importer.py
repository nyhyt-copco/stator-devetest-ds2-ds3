"""Defines pipeline for importing data."""

from pathlib import Path
from typing import Dict

from flux_pipelines.sdk.component.prebuilt import PrebuiltComponent
from kfp import dsl, kubernetes


@dsl.pipeline(
    name="importer",
    description="Pipeline that imports raw data and outputs as samples.",
)
def pipeline(importer_config: Dict):
    """Imports raw data and saves it as samples."""
    base = Path(__file__).parent.parent / "components"
    importer_component = PrebuiltComponent.from_pyproject_toml(base, "importer", input_artifacts=[])

    importer_task = importer_component(config=importer_config)

    kubernetes.add_pod_label(importer_task, "access-ml-pipeline", "true")
    importer_task.set_memory_request("16Gi")
    importer_task.set_memory_limit("32Gi")
    importer_task.set_cpu_request("8")
    importer_task.set_cpu_limit("16")
    importer_task.set_env_variable("PXD_EXECUTOR_BACKEND", "ray")
    importer_task.set_env_variable("PXD_N_CPUS", "8")
