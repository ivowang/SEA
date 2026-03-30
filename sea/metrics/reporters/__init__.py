"""Metrics reporters: console, tensorboard, wandb."""

# Import reporters to register in REPORTER_REGISTRY
import sea.metrics.reporters.console  # noqa: F401
try:
    import sea.metrics.reporters.tensorboard  # noqa: F401
except ImportError:
    pass
try:
    import sea.metrics.reporters.wandb  # noqa: F401
except ImportError:
    pass
