from typing import Literal

from pxc.splitting.split.base import SplitCallable, SplitConfig
from pxd.typing.deferred import DeferredCollection
from pydantic import model_validator
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

    valid_dataset_name: str = "valid"
    invalid_dataset_name: str = "invalid"

    # Optional: if True, invalid samples are not returned as a dataset
    # (useful when you only want the valid split in downstream processing)
    drop_invalid: bool = False

    # Optional: if True, exceptions while reading a sample are raised
    # instead of classifying the sample as invalid
    raise_on_error: bool = False

    @model_validator(mode="after")
    def check_dataset_names_are_different(self) -> Self:
        """Ensure valid and invalid dataset names are different."""
        if self.valid_dataset_name == self.invalid_dataset_name:
            raise ValueError("valid_dataset_name and invalid_dataset_name must be different.")
        return self

    def _is_valid_sample(self, sample: Sample) -> bool:
        """
        Validation logic for a single sample.

        A sample is valid if:
        - 'heatflux_matrix' exists in sample metadata.datablocks
        - flags['missing_heatflux'] is not True
        """
        try:
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

        except Exception:
            if self.raise_on_error:
                raise
            return False

    def _split(
        self,
        samples: DeferredCollection[Sample],
    ) -> dict[str, DeferredCollection[Sample]]:
        """
        Split samples into valid and invalid datasets.
        """
        valid_samples: DeferredCollection[Sample] = {}
        invalid_samples: DeferredCollection[Sample] = {}

        for sample_id, sample in samples.items():
            if self._is_valid_sample(sample):
                valid_samples[sample_id] = sample
            else:
                invalid_samples[sample_id] = sample

        output: dict[str, DeferredCollection[Sample]] = {
            self.valid_dataset_name: valid_samples,
        }

        if not self.drop_invalid:
            output[self.invalid_dataset_name] = invalid_samples

        return output

    def get_dataset_names(self) -> set[str]:
        """Return the dataset names produced by this splitter."""
        if self.drop_invalid:
            return {self.valid_dataset_name}
        return {self.valid_dataset_name, self.invalid_dataset_name}

    def initialize(self) -> SplitCallable:
        """Initialize the split function."""
        return self._split