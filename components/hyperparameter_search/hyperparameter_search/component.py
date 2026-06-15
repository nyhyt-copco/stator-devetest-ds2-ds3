"""Defines component for hyperparameter search."""

from typing import Any

from custom_modelling import register_modelling_configs
from physicsx.quark import DictMaterializer, component
from pxc.experimental.hyperparameter_search.config import HyperparameterSearchComponentConfig
from pxd.datasets.legacy_samples.directory import LegacySampleDirectoryDataset
from pxd.typing.deferred import DeferredCollection
from samples import Sample

register_modelling_configs()


@component(
    config=DictMaterializer,
    train_samples=LegacySampleDirectoryDataset,
    validation_samples=LegacySampleDirectoryDataset,
)
def hyperparameter_search(
    config: dict[str, Any],
    train_samples: DeferredCollection[Sample],
    validation_samples: DeferredCollection[Sample],
) -> None:
    """Component for hyperparameter search over a collection of samples."""
    hyperparameter_search_config = HyperparameterSearchComponentConfig.model_validate(config)
    hyperparameter_search_config.initialize().execute(train_samples, validation_samples)
