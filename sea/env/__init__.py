"""Environment interface and benchmark adapters."""

from sea.env.base import SEAEnv

# Import adapters to register in ENV_REGISTRY
import sea.env.benchmarks.textcraft  # noqa: F401
try:
    import sea.env.benchmarks.alfworld  # noqa: F401
except ImportError:
    pass  # alfworld not installed
try:
    import sea.env.benchmarks.webshop  # noqa: F401
except ImportError:
    pass  # webshop not installed

__all__ = ["SEAEnv"]
