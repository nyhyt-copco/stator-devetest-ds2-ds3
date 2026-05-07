"""Scatter artifact for creating scatter plots of the ground truth and predicted values."""

from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from pxc.evaluation.artifacts.core.matplotlib import BaseMatplotlibArtifact
from pxd.typing.deferred import DeferredCollection
from samples import Sample


def scatter_plot(
    x,
    y,
    y_std: Union[list, np.ndarray] | None = None,
    title: str = "Scatter Plot",
    xlabel: str = "X values",
    ylabel: str = "Y values",
    figsize: Tuple[float, float] = (8, 8),
    show_diagonal: bool = True,
) -> plt.Figure:
    """Create a scatter plot of the ground truth and predicted values.

    Args:
        x : array-like
            X coordinates of the scatter points
        y : array-like
            Y coordinates of the scatter points
        y_std : array-like, optional
            Standard deviation for y values to plot confidence intervals
        title : str
            Plot title (will be right-aligned)
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        figsize : tuple
            Figure size (width, height)
        show_diagonal : bool
            Whether to show y=x diagonal line

    Returns:
        fig : matplotlib figure
    """
    # Style configuration
    style: Dict[str, Any] = {
        "scatter_color": "#3498db",
        "scatter_alpha": 0.7,
        "scatter_size": 50,
        "scatter_edgecolor": "#2c3e50",
        "scatter_linewidth": 1.0,
        "marker_style": "x",
        "diagonal_color": "#e74c3c",
        "diagonal_alpha": 0.8,
        "diagonal_linewidth": 2,
        "diagonal_linestyle": "--",
        "confidence_color": "#95a5a6",
        "confidence_alpha": 0.3,
        "grid": True,
        "grid_alpha": 0.3,
        "spine_color": "#34495e",
        "background_color": "#ffffff",
        "font_size": 12,
        "title_size": 16,
        "dpi": 100,
        "equal_aspect": True,
    }

    # Convert inputs to numpy arrays
    x = np.array(x)
    y = np.array(y)
    if y_std is not None:
        y_std = np.array(y_std)

    # Create figure and axis with square dimensions
    fig, ax = plt.subplots(figsize=figsize, dpi=style["dpi"])
    fig.patch.set_facecolor(style["background_color"])
    ax.set_facecolor(style["background_color"])

    # Plot confidence intervals if provided
    if y_std is not None:
        ax.fill_between(
            x,
            y - y_std,
            y + y_std,
            color=style["confidence_color"],
            alpha=style["confidence_alpha"],
            label="Confidence Interval",
        )

    # Create scatter plot with X markers
    ax.scatter(
        x,
        y,
        c=style["scatter_color"],
        alpha=style["scatter_alpha"],
        s=style["scatter_size"],
        marker=style["marker_style"],
        linewidths=style["scatter_linewidth"],
        edgecolors=style["scatter_edgecolor"],
        zorder=3,
    )

    # Add y=x diagonal line if requested
    if show_diagonal:
        # Get axis limits to draw diagonal across the full range
        all_values = np.concatenate([x, y])
        min_val = np.min(all_values)
        max_val = np.max(all_values)

        # Add some padding
        range_pad = (max_val - min_val) * 0.05
        min_val -= range_pad
        max_val += range_pad

        diagonal_x = np.linspace(min_val, max_val, 100)
        ax.plot(
            diagonal_x,
            diagonal_x,
            color=style["diagonal_color"],
            alpha=style["diagonal_alpha"],
            linewidth=style["diagonal_linewidth"],
            linestyle=style["diagonal_linestyle"],
            label="y = x",
            zorder=2,
        )

    # Set equal aspect ratio to make the plot square in data space
    if style["equal_aspect"]:
        ax.set_aspect("equal", adjustable="box")

    # Add right-aligned title
    ax.text(
        0.98,
        1.02,
        title,
        transform=ax.transAxes,
        fontsize=style["title_size"],
        color=style["spine_color"],
        fontweight="light",
        horizontalalignment="right",
        verticalalignment="bottom",
    )

    # Customize appearance
    ax.set_xlabel(
        xlabel, fontsize=style["font_size"], color=style["spine_color"], fontweight="light"
    )
    ax.set_ylabel(
        ylabel, fontsize=style["font_size"], color=style["spine_color"], fontweight="light"
    )

    # Style the spines and ticks
    for spine in ax.spines.values():
        spine.set_color(style["spine_color"])
        spine.set_linewidth(1.2)
    ax.tick_params(colors=style["spine_color"], which="both", labelsize=style["font_size"] - 1)

    # Add grid if requested
    if style["grid"]:
        ax.grid(True, alpha=style["grid_alpha"], linestyle="-", linewidth=0.8)
        ax.set_axisbelow(True)

    # Add legend if there are multiple elements
    if y_std is not None or show_diagonal:
        legend = ax.legend(loc="best", framealpha=0.9, fontsize=style["font_size"] - 1)
        legend.get_frame().set_facecolor(style["background_color"])
        legend.get_frame().set_edgecolor(style["spine_color"])

    # Adjust layout to accommodate title
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for title

    return fig


class ExampleScatter(BaseMatplotlibArtifact):
    """Creates a scatter plot of the ground truth and predicted values."""

    def __init__(
        self,
        name: str,
        datasets: list[str],
        title: str,
        x: str,
        y: str,
        x_label: str | None = None,
        y_label: str | None = None,
        y_std: str | None = None,
        figsize: Tuple[float, float] = (8, 8),
        show_diagonal: bool = True,
    ):
        """Initialize the Scatter artifact."""
        super().__init__(name, datasets)
        self.title = title
        self.x = x
        self.y = y
        self.x_label = x_label or x
        self.y_label = y_label or y
        self.y_std = y_std
        self.figsize = figsize
        self.show_diagonal = show_diagonal

    def plot(self, dataset: DeferredCollection[Sample]) -> plt.Figure:
        """Create a scatter plot of the ground truth and predicted values."""
        x_vals = np.array([sample()[self.x] for sample in dataset.values()])
        y_vals = np.array([sample()[self.y] for sample in dataset.values()])

        # Get confidence intervals if specified
        y_std_vals = None
        if self.y_std is not None:
            y_std_vals = np.array([sample()[self.y_std] for sample in dataset.values()])

        # Use the scatter_plot function
        fig = scatter_plot(
            x=x_vals,
            y=y_vals,
            y_std=y_std_vals,
            title=self.title,
            xlabel=self.x_label,
            ylabel=self.y_label,
            figsize=self.figsize,
            show_diagonal=self.show_diagonal,
        )

        return fig
