"""Config class for the Scatter artifact."""

from typing import Literal

from pxc.evaluation.artifacts.config.base import ArtifactConfig

from ..core.example_scatter import ExampleScatter


class ExampleScatterConfig(ArtifactConfig):
    """Config class for the Scatter artifact."""

    type: Literal["example_scatter"] = "example_scatter"
    datasets: list[str]
    title: str
    x: str
    y: str
    x_label: str | None = None
    y_label: str | None = None
    y_std: str | None = None
    figsize: tuple[float, float] = (8, 8)
    show_diagonal: bool = True

    def initialize(self) -> "ExampleScatter":
        """Initialize the Scatter artifact."""
        return ExampleScatter(
            name=self.name,
            datasets=self.datasets,
            title=self.title,
            x=self.x,
            y=self.y,
            x_label=self.x_label,
            y_label=self.y_label,
            y_std=self.y_std,
            figsize=self.figsize,
            show_diagonal=self.show_diagonal,
        )
