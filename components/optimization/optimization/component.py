"""Defines optimization component."""

import logging
import os
from typing import Annotated, Any

import pandas as pd
from flux import Flux
from flux.common.credentials import Credentials
from flux.shared_services import cred_mgr
from physicsx.quark import DictMaterializer, component
from pxc.optimization.config import OptimizationConfig
from pxd.datasets.legacy_samples.directory import LegacySampleDirectoryDataset
from pxd.datasets.pandas.csv.file import PandasCSVFileDataset
from pxd.typing.deferred import DeferredCollection
from samples import Sample

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PAT_TOKEN_NAME = "PLATFORM_API_PAT_TOKEN_NAME"
PAT_TOKEN_VALUE = "PLATFORM_API_PAT_TOKEN_VALUE"
TENANT = "PX_TENANT"


@component(
    optimization_config=DictMaterializer,
    train_samples=LegacySampleDirectoryDataset,
    optimization_results=PandasCSVFileDataset,
)
def optimization(
    optimization_config: dict[str, Any],
    train_samples: DeferredCollection[Sample],
) -> Annotated[pd.DataFrame, "optimization_results"]:
    """Component for optimization process."""
    config = OptimizationConfig.model_validate(optimization_config)

    if config.log_to_optimizer:
        authenticate_flux()

    optimization_results = config.initialize().execute(train_samples)

    return optimization_results


def authenticate_flux() -> None:
    """Attempts to authenticate with Flux using existing credentials.

    If that fails, try to store credentials from PAT tokens.
    """
    try:
        _ = Flux()
        logger.info("Flux instantiation successful with existing credentials.")
    except Exception as e:
        logger.warning(f"Flux instantiation failed: {e}")
        logger.info("Attempting to store credentials from PAT tokens instead...")

        store_flux_credentials()


def store_flux_credentials() -> None:
    """Gets Flux Credentials and stores them in the shared credential manager.

    Ensures they are available for future Flux() calls.
    """
    pat_name = os.getenv(PAT_TOKEN_NAME)
    pat_token = os.getenv(PAT_TOKEN_VALUE)
    tenant = os.getenv(TENANT)

    logger.info(f"PAT Name present: {bool(pat_name)}")
    logger.info(f"PAT Token present: {bool(pat_token)}")
    logger.info(f"Tenant: {tenant}")

    if not (bool(pat_name) and bool(pat_token)):
        raise ValueError(
            f"PAT tokens not set in tenant: {tenant}. "
            "Either set log_to_optimizer=False to run without logging, "
            f"or ask for PAT tokens to be set in: {tenant}"
        )

    credentials = Credentials(
        organization=tenant,
        pat_name=pat_name,
        pat_token=pat_token,
    )

    if not credentials.is_valid():
        raise ValueError("Invalid credentials provided")

    cred_mgr.set_token_credentials(credentials, emit_logout_event=False)

    try:
        _ = Flux()
        logger.info("Flux instantiation successful after storing credentials.")
    except Exception as e:
        raise ValueError(f"Failed to instantiate Flux after storing credentials: {e}")
