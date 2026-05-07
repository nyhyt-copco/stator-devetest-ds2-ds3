"""Registry of scheduler classes."""

from .config.power_decay_one_cycle import ExamplePowerDecayOneCycleSchedulerConfig

# Registry of scheduler classes
scheduler_registry = [
    ExamplePowerDecayOneCycleSchedulerConfig,
]
