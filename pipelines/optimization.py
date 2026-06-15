"""Defines pipeline for optimization."""

import os
from pathlib import Path
from typing import Dict

from flux_pipelines.sdk.component.prebuilt import PrebuiltComponent
from kfp import dsl, kubernetes

ENVS_TO_MOUNT = [
    "PX_TENANT",
    "MLFLOW_SERVICE_HTTP_URL",
    "FLUX_CLI_API_ENDPOINT",
    "FLUX_CLI_TOKEN_EXCHANGE_ENDPOINT",
]

PX_TENANT = os.getenv("PX_TENANT")
PX_RING = os.getenv("PX_RING")


@dsl.pipeline(
    name="optimization",
    description="Pipeline that runs optimization and outputs results.",
)
def pipeline(optimization_config: Dict):
    """Runs optimization and outputs results."""
    base = Path(__file__).parent.parent / "components"
    optimization_component = PrebuiltComponent.from_pyproject_toml(
        base, "optimization", input_artifacts=[]
    )

    optimization_task = optimization_component(config=optimization_config)
    kubernetes.add_pod_label(optimization_task, "access-ml-pipeline", "true")
    optimization_task.set_memory_request("8Gi")
    optimization_task.set_memory_limit("16Gi")
    optimization_task.set_cpu_request("2")
    optimization_task.set_cpu_limit("4")
    optimization_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")

    kubernetes.use_secret_as_env(
        optimization_task,
        secret_name=f"{PX_TENANT}-{PX_RING}-aiwb-api-system-personal-access-token",
        secret_key_to_env={
            "personal_access_token_name": "PLATFORM_API_PAT_TOKEN_NAME",
            "personal_access_token_value": "PLATFORM_API_PAT_TOKEN_VALUE",
        },
    )

    # Mount additional environment variables
    for env in ENVS_TO_MOUNT:
        optimization_task.set_env_variable(env, os.getenv(env))
