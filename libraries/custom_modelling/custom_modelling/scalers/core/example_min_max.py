"""Defines MinMax scaler."""

import torch
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from pxs.distribution.distribution import Distribution
from pxs.scalers.sklearn import SklearnScaler


class ExampleMinMaxScaler(SklearnScaler):
    """MinMax Scaler (uses Sklearn)."""

    def __init__(self, range_min: float = 0, range_max: float = 1) -> None:
        """Constructor for the MinMaxScaler class.

        Args:
            range_min: range minimum
            range_max: range maximum
        """
        scaler = SklearnMinMaxScaler(feature_range=(range_min, range_max))
        super().__init__(
            scaler=scaler,
            support_sample_weight=False,
        )
        self._scaler: SklearnMinMaxScaler

    @property
    def _min(self) -> torch.Tensor:
        """Min of the scaler.

        Returns:
            Min of the scaler as a torch tensor.
        """
        return torch.from_numpy(self._scaler.min_)

    @property
    def _scale(self) -> torch.Tensor:
        """Scale of the scaler.

        Returns:
            Scale of the scaler as a torch tensor.
        """
        return torch.from_numpy(self._scaler.scale_)

    def _transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply scaling.

        Args:
            data: data to scale, shape: [n_samples, n_features]

        Returns:
            Scaled data.
        """
        original_shape = data.shape
        data = data * self._scale.to(data.device, dtype=data.dtype)
        data = data + self._min.to(data.device, dtype=data.dtype)
        return data.reshape(original_shape)

    def _inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Revert scaling.

        Args:
            data: data to un-scale, shape: [n_samples, n_features]

        Returns:
            Un-scaled data.
        """
        original_shape = data.shape
        data = data - self._min.to(data.device, dtype=data.dtype)
        data = data / self._scale.to(data.device, dtype=data.dtype)
        return data.reshape(original_shape)

    def _transform_distribution(self, data: Distribution) -> Distribution:
        """Apply scaling to distribution.

        Args:
            data: distribution to scale.

        Returns:
            Scaled distribution.
        """
        bias = self._min.to(data.device, dtype=data.dtype)
        scale = self._scale.to(data.device, dtype=data.dtype)
        return data.linear_scale(scale, bias)

    def _inverse_transform_distribution(self, data: Distribution) -> Distribution:
        """Revert scaling to distribution.

        Args:
            data: distribution to scale.

        Returns:
            Scaled distribution.
        """
        bias = self._min.to(data.device, dtype=data.dtype)
        scale = self._scale.to(data.device, dtype=data.dtype)

        return data.linear_scale(1 / scale, -bias / scale)
