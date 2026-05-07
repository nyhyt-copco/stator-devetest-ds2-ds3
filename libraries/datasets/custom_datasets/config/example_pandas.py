"""Defines config for an example Pandas dataset type."""

from typing import Any, Literal

from pxd.experimental.datasets.config.base.dataset import DatasetConfig
from pxd.experimental.typing.data import SupportedPandasFormats

from ..core.example_pandas import ExamplePandasDataset


class ExamplePandasDatasetConfig(DatasetConfig):
    """Example config for loading a directory of CSV or Parquet files.

    Demonstrates how to register a custom dataset type that can be referenced
    in importer YAML configuration via ``type: example_pandas_dataset``.
    """

    type: Literal["example_pandas_dataset"] = "example_pandas_dataset"

    load_args: dict[str, Any] | None = None
    save_args: dict[str, Any] | None = None

    format: SupportedPandasFormats | Literal["infer"] = "infer"

    replace_values: dict[str, Any] | None = None
    replace_column_values: dict[str, Any] | None = None
    column_dtypes: dict[str, str] | None = None
    transpose: bool = False
    remove_nans: bool = False

    def initialize(self) -> ExamplePandasDataset:
        """Initialize the ExamplePandasDataset.

        Returns:
            ExamplePandasDataset: Initialized dataset instance.
        """
        return ExamplePandasDataset(
            path=self.get_path(),
            name=self.name,
            glob=self.glob,
            pattern=self.pattern,
            load_args=self.load_args,
            save_args=self.save_args,
            format=self.format,
            overwrite=self.overwrite,
            filesystem=self.filesystem,
            handle_empty_dataset=self.handle_empty_dataset,
            handle_duplicate_ids=self.handle_duplicate_ids,
            replace_values=self.replace_values,
            replace_column_values=self.replace_column_values,
            column_dtypes=self.column_dtypes,
            transpose=self.transpose,
            remove_nans=self.remove_nans,
        )
