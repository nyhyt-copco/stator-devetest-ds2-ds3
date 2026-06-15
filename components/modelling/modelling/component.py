"""Defines component for modelling samples."""

from typing import Annotated, Any

from custom_modelling import register_modelling_configs
from pxc.modelling.config import ModellingConfig
from pxd.typing.deferred import DeferredCollection
from samples import Sample
from physicsx.quark import component, DictMaterializer
from pxd.datasets.legacy_samples.directory import LegacySampleDirectoryDataset

register_modelling_configs()


@component(
    config=DictMaterializer,
    train_samples=LegacySampleDirectoryDataset,
    validation_samples=LegacySampleDirectoryDataset,
    test_samples=LegacySampleDirectoryDataset,
    train_predictions=LegacySampleDirectoryDataset,
    validation_predictions=LegacySampleDirectoryDataset,
    test_predictions=LegacySampleDirectoryDataset,
)
def modelling(
    config: dict[str, Any],
    train_samples: DeferredCollection[Sample],
    validation_samples: DeferredCollection[Sample],
    test_samples: DeferredCollection[Sample],
) -> tuple[
    Annotated[DeferredCollection[Sample], "train_predictions"],
    Annotated[DeferredCollection[Sample], "validation_predictions"],
    Annotated[DeferredCollection[Sample], "test_predictions"],
]:
    """Component for modeling a collection of samples."""
    validated_config = ModellingConfig.model_validate(config)

    _, train_predictions, validation_predictions, test_predictions = (
        validated_config.initialize().execute(train_samples, validation_samples, test_samples)
    )

    return train_predictions, validation_predictions, test_predictions
