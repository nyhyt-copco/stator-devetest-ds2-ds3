from typing import Annotated, Literal
from math import isclose
import numpy as np
from pxc.splitting.split.base import SplitCallable, SplitConfig
from pxd.typing.deferred import DeferredCollection
from pydantic import Field, model_validator
from samples import Sample
from typing_extensions import Self


class HeatfluxValiditySplitConfig(SplitConfig):
    """
    Split samples into valid / invalid datasets based on:
      1. presence of 'heatflux_matrix' in metadata.datablocks
      2. absence of flags['missing_heatflux'] == True

    This is config-friendly and can be used directly from YAML / config files.
    """

    split_type: Literal["heatflux_validity"] = "heatflux_validity"

    datasets: dict[str, Annotated[float, Field(gt=0, le=1)]]

    random_state: int = 42

    @model_validator(mode="after")
    def check_train_split_exists(self) -> Self:
        """Check if 'train' key exists in the dataset splits."""
        if "train" not in self.datasets:
            raise KeyError("Dataset splits missing 'train' split.")

        return self

    @model_validator(mode="after")
    def check_sum_proportions_less_than_one(self) -> Self:
        """Check if sum of proportions is less than 1."""
        total = sum(self.datasets.values())

        if not isclose(total, 1):
            raise ValueError(f"Sum of proportions must equal 1. Got: {total}")

        return self

    def _is_valid_sample(self, sample: Sample) -> bool:
        """
        Validation logic for a single sample.

        A sample is valid if:
        - 'heatflux_matrix' exists in sample metadata.datablocks
        - flags['missing_heatflux'] is not True
        """
        
        sample_dict = sample.to_dict()
        datablocks = sample_dict.get("metadata", {}).get("datablocks", {})

        # Check 1: heatflux_matrix exists
        if "heatflux_matrix" not in datablocks:
            return False

        # Check 2: flags block
        flags_block = sample.get("flags")
        if flags_block is not None:
            flags = flags_block.read()

            # If flags is dict-like and says missing_heatflux=True => invalid
            if isinstance(flags, dict) and flags.get("missing_heatflux") is True:
                return False

        return True


    def _split(self, samples: DeferredCollection[Sample]) -> dict[str, DeferredCollection[Sample]]:
        dataset_names, dataset_sizes = zip(*self.datasets.items())

        # Keep only valid samples first
        valid_samples: DeferredCollection[Sample] = {
            sample_id: sample
            for sample_id, sample in samples.items()
            if self._is_valid_sample(sample())
        }

        valid_sample_ids = list(valid_samples.keys())

        # Randomly permute only valid sample IDs
        permutation = np.random.default_rng(self.random_state).permutation(valid_sample_ids)

        # Compute section boundaries based on number of valid samples
        sections = np.cumsum([size * len(valid_samples) for size in dataset_sizes]).round().astype(int)

        # np.split expects split points before the final length
        splits = np.split(permutation, sections[:-1])

        samples_by_dataset: dict[str, DeferredCollection[Sample]] = {}

        for name, sample_ids in zip(dataset_names, splits):
            samples_by_dataset[name] = {
                sample_id: valid_samples[sample_id]
                for sample_id in sample_ids
            }

        return samples_by_dataset


    def get_dataset_names(self) -> set[str]:
        """Return the names of all datasets."""
        return set(self.datasets.keys())

    def initialize(self) -> SplitCallable:
        """Initialize the split function."""
        return self._split