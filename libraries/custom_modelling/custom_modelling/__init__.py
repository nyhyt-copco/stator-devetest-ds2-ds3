"""Modelling package."""

from .losses.config import register_loss_configs
from .metrics.config import register_metric_configs
from .models.config import register_model_configs
from .optimizers.config import register_optimizer_configs
from .scalers.config import register_scaler_configs
from .schedulers.config import register_scheduler_configs


def register_modelling_configs():
    """Register all modelling configs."""
    register_loss_configs()
    register_metric_configs()
    register_model_configs()
    register_optimizer_configs()
    register_scaler_configs()
    register_scheduler_configs()


__all__ = [
    "register_modelling_configs",
    "register_loss_configs",
    "register_metric_configs",
    "register_model_configs",
    "register_optimizer_configs",
    "register_scaler_configs",
    "register_scheduler_configs",
]
