"""Defines node for creating the dataset sample before modelling."""

from math import isclose
from typing import Annotated, Literal

import numpy as np
from pxc.splitting.split.base import SplitCallable, SplitConfig
from pxd.typing.deferred import DeferredCollection
from pydantic import Field, model_validator
from samples import Sample
from typing_extensions import Self


class ExampleRandomSplitConfig(SplitConfig):
    """Kedro config for splitting samples into datasets."""

    split_type: Literal["example_random"] = "example_random"
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

    def _split(self, samples: DeferredCollection[Sample]) -> dict[str, DeferredCollection[Sample]]:
        dataset_names, dataset_sizes = zip(*self.datasets.items())

        permutation = np.random.default_rng(self.random_state).permutation(list(samples.keys()))
        sections = np.cumsum([size * len(samples) for size in dataset_sizes]).round().astype(int)

        splits = np.split(permutation, sections)
        samples_by_dataset: dict[str, DeferredCollection[Sample]] = {}

        for name, sample_ids in zip(dataset_names, splits):
            samples_by_dataset[name] = {sample_id: samples[sample_id] for sample_id in sample_ids}

        return samples_by_dataset

    def get_dataset_names(self) -> set[str]:
        """Return the names of all datasets."""
        return set(self.datasets.keys())

    def initialize(self) -> SplitCallable:
        """Initialize the random split function."""
        return self._split
