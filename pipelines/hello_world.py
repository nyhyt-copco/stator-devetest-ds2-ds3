"""Defines a simple pipeline that runs on Kubeflow."""

from pathlib import Path
from typing import Dict

from flux_pipelines.sdk.component.prebuilt import PrebuiltComponent
from kfp import dsl


@dsl.pipeline(
    name="hello_world", description="Prints 'Hello ${ config.name }!' in a custom component."
)
def pipeline(hello_world_config: Dict):
    """A simple hello world pipeline."""
    base = Path(__file__).parent.parent / "components"
    hello_world_component = PrebuiltComponent.from_pyproject_toml(base, "hello_world")

    hello_world_component(config=hello_world_config)
