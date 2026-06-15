"""Registry of split classes."""

from .config.example_random import ExampleRandomSplitConfig
from .config.heatFluxValidatySplitConfig import HeatfluxValiditySplitConfig

# Registry of split classes
split_registry = [
    ExampleRandomSplitConfig,
    HeatfluxValiditySplitConfig
]
