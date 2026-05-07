"""Defines hello component."""

from typing import Any

from physicsx.quark import component, DictMaterializer
from pydantic import BaseModel, Field
from commons.main import say_hello


class HelloWorldConfig(BaseModel):
    """Config for Hello parameters."""

    name: str = Field(default="World")


@component(config=DictMaterializer)
def hello_world(
    config: dict[str, Any],
) -> None:
    """Prints a hello message."""
    validated_config = HelloWorldConfig.model_validate(config)
    message = say_hello(validated_config.name)
    print(message)
