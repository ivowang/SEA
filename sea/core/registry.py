"""Lightweight component registry for plugin discovery.

Researchers register custom implementations with ``@REGISTRY.register("name")``
and reference them by name in YAML configs.
"""

from __future__ import annotations

from typing import Any, TypeVar, Type

T = TypeVar("T")


class Registry(dict[str, Type]):
    """A named registry that maps string keys to classes.

    Usage::

        PLANNER_REGISTRY = Registry("planner")

        @PLANNER_REGISTRY.register("react")
        class ReActPlanner(Planner):
            ...

        # Later, build from config:
        planner = PLANNER_REGISTRY.build("react", brain=brain)
    """

    def __init__(self, name: str = "") -> None:
        super().__init__()
        self.registry_name = name

    def register(self, name: str):
        """Decorator that registers a class under *name*."""
        def decorator(cls: Type[T]) -> Type[T]:
            if name in self:
                raise KeyError(
                    f"Registry '{self.registry_name}': "
                    f"name '{name}' already registered to {self[name]}"
                )
            self[name] = cls
            return cls
        return decorator

    def build(self, name: str, **kwargs: Any):
        """Instantiate the class registered under *name*."""
        if name not in self:
            available = ", ".join(sorted(self.keys())) or "(empty)"
            raise KeyError(
                f"Registry '{self.registry_name}': "
                f"'{name}' not found. Available: {available}"
            )
        return self[name](**kwargs)

    def __repr__(self) -> str:
        entries = ", ".join(sorted(self.keys()))
        return f"Registry('{self.registry_name}', [{entries}])"


# ---------------------------------------------------------------------------
# Global registries
# ---------------------------------------------------------------------------

AGENT_REGISTRY = Registry("agent")
ENV_REGISTRY = Registry("env")
EVOLVER_REGISTRY = Registry("evolver")
PLANNER_REGISTRY = Registry("planner")
MEMORY_REGISTRY = Registry("memory")
SKILL_REGISTRY = Registry("skill")
TOOL_REGISTRY = Registry("tool")
LLM_BACKEND_REGISTRY = Registry("llm_backend")
REWARD_REGISTRY = Registry("reward")
REPORTER_REGISTRY = Registry("reporter")
