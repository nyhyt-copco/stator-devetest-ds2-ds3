"""Defines component for preprocessing samples."""

from typing import Annotated, Any

from custom_layers import register_layer_configs
from physicsx.quark import component, DictMaterializer
from pxc.preprocessing.component import Preprocessing
from pxc.preprocessing.config import PreprocessingConfig
from pxd.datasets.legacy_samples.directory import LegacySampleDirectoryDataset
from pxd.typing.deferred import DeferredCollection
from samples import Sample


register_layer_configs()

@component(
    config=DictMaterializer,
    samples=LegacySampleDirectoryDataset,
    preprocessed_samples=LegacySampleDirectoryDataset,
)
def preprocessing(
    config: dict[str, Any], samples: DeferredCollection[Sample]
) -> Annotated[DeferredCollection[Sample], "preprocessed_samples"]:
    """Component for preprocessing a collection of samples."""
    validated_config = PreprocessingConfig.model_validate(config)
    preprocessing_component = Preprocessing(
        [layer.initialize() for layer in validated_config.layers],
        inplace=validated_config.inplace,
        executor=validated_config.executor,
        n_cpus=validated_config.n_cpus,
        tracking=validated_config.tracking,
    )
    preprocessed_samples = preprocessing_component.execute(samples)
    return preprocessed_samples
