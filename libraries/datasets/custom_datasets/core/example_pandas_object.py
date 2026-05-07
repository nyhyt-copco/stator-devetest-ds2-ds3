"""Defines an example single-file Pandas dataset object."""

from pathlib import Path, PurePath
from typing import Any, Hashable, Literal, Mapping

import numpy as np
import pandas as pd
import samples

from pxd.experimental.datasets.core.base.decorators import (
    download_from_filesystem,
    upload_to_filesystem,
)
from pxd.experimental.datasets.core.base.object import DatasetObject
from pxd.experimental.multidataset.utils import SupportsMultiDataset
from pxd.experimental.typing.data import SupportedPandasFormats
from pxd.filesystems.config import FilesystemClientConfig


class ExamplePandasObject(DatasetObject[pd.DataFrame], SupportsMultiDataset[dict[Hashable, Any]]):
    """Loads and saves a single Pandas DataFrame from a CSV or Parquet file."""

    _supported_suffixes: dict[SupportedPandasFormats, set[str]] = {
        "csv": {".csv"},
        "parquet": {".parquet", ".pq"},
        "fixed-width": {".text", ".txt"},
    }

    def __init__(  # noqa: PLR0913
        self,
        path: PurePath | str,
        name: str | None = None,
        *,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        format: SupportedPandasFormats | Literal["infer"] = "infer",
        overwrite: bool = False,
        filesystem: FilesystemClientConfig | dict[str, Any] | None = None,
        sample_id_column: str = "id",
        replace_values: dict[str, Any] | None = None,
        replace_column_values: dict[str, Any] | None = None,
        column_dtypes: Mapping[str, np.dtype[Any] | str] | None = None,
        remove_nans: bool = False,
    ) -> None:
        """Initialize the ExamplePandasObject.

        Args:
            path: Path to the file.
            name: Name of the dataset object.
            load_args: Keyword arguments passed to the pandas reader function.
            save_args: Keyword arguments passed to the pandas writer function.
            format: Reader/writer format. Accepts "csv", "parquet", "fixed-width", "infer".
            overwrite: Allow overwriting the file.
            filesystem: Filesystem configuration.
            sample_id_column: Column name to match to Sample IDs.
            replace_values: Values to replace in the DataFrame when adding to Samples.
            replace_column_values: Column name replacements when adding to Samples.
            column_dtypes: Column dtype conversions when adding to Samples.
            remove_nans: Drop rows with NaN values when adding to Samples.
        """
        super().__init__(path=path, name=name, overwrite=overwrite, filesystem=filesystem)

        self.load_args = load_args if load_args is not None else {}
        self.save_args = save_args if save_args is not None else {}
        self.format: SupportedPandasFormats | Literal["infer"] = format

        self.sample_id_column = sample_id_column
        self.replace_values = replace_values
        self.replace_column_values = replace_column_values
        self.column_dtypes = column_dtypes
        self.remove_nans = remove_nans

    def _get_format(self, filepath: PurePath) -> SupportedPandasFormats:
        """Determine the file format from its suffix, or use the configured format."""
        if self.format != "infer":
            return self.format

        for fmt, suffixes in self._supported_suffixes.items():
            if filepath.suffix in suffixes:
                return fmt

        raise RuntimeError(f"Could not determine the format for '{filepath.suffix}'.")

    @download_from_filesystem
    def load(self, local_filepath: Path) -> pd.DataFrame:
        """Load a DataFrame from the file."""
        fmt = self._get_format(local_filepath)

        if fmt == "csv":
            return pd.read_csv(local_filepath, **self.load_args)
        if fmt == "parquet":
            return pd.read_parquet(local_filepath, **self.load_args)
        if fmt == "fixed-width":
            return pd.read_fwf(local_filepath, **self.load_args)

        raise ValueError(f"Format '{fmt}' is not supported.")

    @upload_to_filesystem
    def save(self, local_filepath: Path, data: pd.DataFrame) -> None:
        """Save a DataFrame to the file."""
        fmt = self._get_format(local_filepath)

        if fmt == "csv":
            data.to_csv(local_filepath, **self.save_args)
            return
        if fmt == "parquet":
            data.to_parquet(local_filepath, **self.save_args)
            return
        if fmt == "fixed-width":
            raise NotImplementedError("Writing fixed-width files is not supported.")

        raise ValueError(f"Unknown format '{fmt}'.")

    def load_sample_mapping(self) -> dict[str, dict[Hashable, Any]]:
        """Load the mapping from sample ID to data from the file."""
        df = self.load()

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

        return {str(record.pop(self.sample_id_column)): record for record in df.to_dict("records")}

    def add_data_to_sample_v2(
        self, sample: samples.Sample, name: str, data: dict[Hashable, Any]
    ) -> samples.Sample:
        """Add data from a DataFrame row to a Sample object."""
        arrays = {}
        for key, value in data.items():
            if isinstance(value, str):
                value_array = np.array(value.encode("utf-8"), dtype="S")
            else:
                value_array = np.array(value)
            arrays[key] = value_array

        sample.add(name, arrays, "scalar_data")
        return sample
