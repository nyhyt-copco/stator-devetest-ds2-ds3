"""Defines component for importing data as samples."""

from typing import Annotated, Any

from custom_datasets import register_dataset_configs
from physicsx.quark import DictMaterializer, component
from pxc.importer.config import ImporterConfig
from pxd.datasets.legacy_samples.directory import LegacySampleDirectoryDataset
from pxd.typing.deferred import DeferredCollection
from samples import Sample

register_dataset_configs()


@component(config=DictMaterializer, samples=LegacySampleDirectoryDataset)
def importer(
    config: dict[str, Any],
) -> Annotated[DeferredCollection[Sample], "samples"]:
    """Imports data as VTKs, CSVs, etc. and converts them to v1 samples."""
    validated_config = ImporterConfig.model_validate(config)
    samples = validated_config.initialize().execute()

    return samples
