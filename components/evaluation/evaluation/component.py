"""Defines component for evaluating samples with a model."""

from typing import Annotated, Any

from custom_artifacts import register_artifact_configs
from custom_layers import register_layer_configs
from pxc.evaluation.component import Evaluation
from pxc.evaluation.config import EvaluationConfig
from pxd.typing.deferred import DeferredCollection
from samples import Sample
from physicsx.quark import component, DictMaterializer
from pxd.datasets.legacy_samples.directory import LegacySampleDirectoryDataset

register_layer_configs()
register_artifact_configs()


@component(
    config=DictMaterializer,
    train_samples=LegacySampleDirectoryDataset,
    validation_samples=LegacySampleDirectoryDataset,
    test_samples=LegacySampleDirectoryDataset,
    train_evaluated_samples=LegacySampleDirectoryDataset,
    validation_evaluated_samples=LegacySampleDirectoryDataset,
    test_evaluated_samples=LegacySampleDirectoryDataset,
)
def evaluation(
    config: dict[str, Any],
    train_samples: DeferredCollection[Sample],
    validation_samples: DeferredCollection[Sample],
    test_samples: DeferredCollection[Sample],
) -> tuple[
    Annotated[DeferredCollection[Sample], "train_evaluated_samples"],
    Annotated[DeferredCollection[Sample], "validation_evaluated_samples"],
    Annotated[DeferredCollection[Sample], "test_evaluated_samples"],
]:
    """Component for evaluation a collection of samples."""
    samples = {
        "train": train_samples,
        "validation": validation_samples,
        "test": test_samples,
    }

    validated_config = EvaluationConfig.model_validate(config)

    evaluation_component = Evaluation(
        [layer.initialize() for layer in validated_config.post_processing_layers],
        [artifact.initialize() for artifact in validated_config.artifacts],
        executor=validated_config.executor,
        n_cpus=validated_config.n_cpus,
        tracking=validated_config.tracking,
    )

    post_processed_samples = evaluation_component.execute(samples)

    return (
        post_processed_samples["train"],
        post_processed_samples["validation"],
        post_processed_samples["test"],
    )
