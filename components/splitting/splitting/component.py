"""Defines component for splitting samples."""

from typing import Annotated, Any

from custom_splits import register_split_configs
from physicsx.quark import DictMaterializer, component
from pxc.splitting.component import Splitting
from pxc.splitting.config import SplittingConfig
from pxd.datasets.legacy_samples.directory import LegacySampleDirectoryDataset
from pxd.typing.deferred import DeferredCollection
from samples import Sample

register_split_configs()


@component(
    config=DictMaterializer,
    samples=LegacySampleDirectoryDataset,
    train_samples=LegacySampleDirectoryDataset,
    validation_samples=LegacySampleDirectoryDataset,
    test_samples=LegacySampleDirectoryDataset,
)
def splitting(
    config: dict[str, Any],
    samples: DeferredCollection[Sample],
) -> tuple[
    Annotated[
        DeferredCollection[Sample],
        "train_samples",
    ],
    Annotated[
        DeferredCollection[Sample],
        "validation_samples",
    ],
    Annotated[
        DeferredCollection[Sample],
        "test_samples",
    ],
]:
    """Component for splitting a collection of samples."""
    validated_config = SplittingConfig.model_validate(config)
    split_component = Splitting(
        splits=[split.initialize() for split in validated_config.batch_splits],
        executor=validated_config.executor,
        n_cpus=validated_config.n_cpus,
        tracking=validated_config.tracking,
    )

    splits = split_component.execute(samples)
    return splits["train"], splits["validation"], splits["test"]
