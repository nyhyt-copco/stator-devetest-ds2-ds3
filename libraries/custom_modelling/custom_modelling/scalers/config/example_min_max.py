"""Min Max Scaler Config class definition."""

from typing import Literal

from pxs.scalers.config.base import BaseScalerConfig
from ..core.example_min_max import ExampleMinMaxScaler


class ExampleMinMaxScalerConfig(BaseScalerConfig):
    """Config representing a Min Max scaler.

    The scaler modifies features to a given range by linearly mapping the data to the range
    `[0, 1]` or `[range_min, range_max]`. This transformation is useful when you want to ensure
    that all features have the same scale, making it easier for algorithms to learn.

    The MinMaxScaler is particularly effective for algorithms that assume features have the same
    scale, such as linear regression, logistic regression, and neural networks. You can control the
    range of the scaled features with `range_min` and `range_max`.

    More details could be found in [Scikit Learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).
    """

    scaler_type: Literal["example_min_max"] = "example_min_max"

    range_min: float = 0
    range_max: float = 1

    def initialize(self) -> ExampleMinMaxScaler:
        """Initialize scaler using config.

        Returns:
            Initialized scaler object.
        """
        return ExampleMinMaxScaler(
            range_min=self.range_min,
            range_max=self.range_max,
        )
