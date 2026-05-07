"""Defines an example Pandas collection dataset."""

from pathlib import PurePath
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd
import samples

from pxd.experimental.datasets.core.base.dataset import Dataset
from pxd.experimental.multidataset.utils import SupportsMultiDataset
from pxd.experimental.typing.data import SupportedPandasFormats
from pxd.typing.utils import RaiseWarnIgnore
from pxd.filesystems.config import FilesystemClientConfig
from pxd.typing.deferred import Deferred, DeferredCollection

from .example_pandas_object import ExamplePandasObject


class ExamplePandasDataset(Dataset[pd.DataFrame], SupportsMultiDataset[Deferred[pd.DataFrame]]):
    """Loads a directory of CSV or Parquet files as Pandas DataFrames.

    Each file in the directory becomes one entry in the collection, keyed by a sample ID
    extracted from the file path using the ``pattern`` regex.
    """

    _default_format: SupportedPandasFormats = "csv"
    _suffix = None

    _suffixes: dict[SupportedPandasFormats, str] = {
        "csv": ".csv",
        "parquet": ".pq",
        "fixed-width": ".txt",
    }

    def __init__(  # noqa: PLR0913
        self,
        path: PurePath | str,
        name: str | None = None,
        glob: str | None = None,
        *,
        pattern: str | None = None,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        format: SupportedPandasFormats | Literal["infer"] = "infer",
        overwrite: bool = False,
        filesystem: FilesystemClientConfig | dict[str, Any] | None = None,
        handle_empty_dataset: RaiseWarnIgnore = "raise",
        handle_duplicate_ids: RaiseWarnIgnore = "raise",
        replace_values: dict[str, Any] | None = None,
        replace_column_values: dict[str, Any] | None = None,
        column_dtypes: Mapping[str, np.dtype | str] | None = None,
        transpose: bool = False,
        remove_nans: bool = False,
    ) -> None:
        """Initialize the ExamplePandasDataset.

        Args:
            path: Path to the directory containing the collection.
            name: Name of the dataset.
            glob: Glob pattern to filter files with.
            pattern: Regex pattern to extract sample IDs from file paths.
            load_args: Keyword arguments passed to the pandas reader function.
            save_args: Keyword arguments passed to the pandas writer function.
            format: Reader/writer format. Accepts "csv", "parquet", "fixed-width", "infer".
            overwrite: Allow overwriting existing directories.
            filesystem: Filesystem configuration (local or S3).
            handle_empty_dataset: Behavior when loading an empty collection.
            handle_duplicate_ids: Behavior when finding duplicate sample IDs.
            replace_values: Values to replace in the DataFrame when adding to Samples.
            replace_column_values: Column name replacements when adding to Samples.
            column_dtypes: Column dtype conversions when adding to Samples.
            transpose: Transpose the DataFrame when adding to Samples.
            remove_nans: Drop rows with NaN values when adding to Samples.
        """
        super().__init__(
            path=path,
            name=name,
            glob=glob,
            pattern=pattern,
            overwrite=overwrite,
            filesystem=filesystem,
            handle_empty_dataset=handle_empty_dataset,
            handle_duplicate_ids=handle_duplicate_ids,
        )

        self.load_args = load_args
        self.save_args = save_args
        self.format: SupportedPandasFormats | Literal["infer"] = format

        self.replace_column_values = replace_column_values
        self.replace_values = replace_values
        self.column_dtypes = column_dtypes
        self.transpose = transpose
        self.remove_nans = remove_nans

    def _get_path_from_id(self, object_id: str) -> PurePath:
        """Add the correct file suffix based on the selected format."""
        path = super()._get_path_from_id(object_id)

        format: SupportedPandasFormats
        if self.format != "infer":
            format = self.format
        else:
            format = self._default_format

        return path.with_suffix(self._suffixes[format])

    def _create_dataset(self, path: PurePath) -> ExamplePandasObject:
        """Create an ExamplePandasObject for a single file in the collection."""
        return ExamplePandasObject(
            name=self.name,
            path=path,
            load_args=self.load_args,
            save_args=self.save_args,
            format=self.format,
            overwrite=self.overwrite,
            filesystem=self.filesystem_config,
        )

    def load_sample_mapping(self) -> DeferredCollection[pd.DataFrame]:
        """Return the mapping from sample ID to DataFrame."""
        return self.load()

    def add_data_to_sample_v2(
        self,
        sample: samples.Sample,
        name: str,
        data: Deferred[pd.DataFrame],
    ) -> samples.Sample:
        """Add the DataFrame to a sample as scalar data."""
        df = data()

        if self.transpose:
            df = df.T
            df.reset_index(inplace=True, drop=True)
            df.columns = df.columns.astype(str)

        if self.replace_values is not None:
            df.replace(**self.replace_values, inplace=True)

        if self.replace_column_values is not None:
            df.columns = df.columns.str.replace(**self.replace_column_values)

        if self.remove_nans:
            df.dropna(inplace=True)

        if self.column_dtypes is not None:
            df = df.astype(
                {
                    column: np.dtype(dtype) if isinstance(dtype, str) else dtype
                    for column, dtype in self.column_dtypes.items()
                },
            )

        df_dict = {column: df[column].to_numpy() for column in df.columns}
        sample.add(name, df_dict, "scalar_data")

        return sample
