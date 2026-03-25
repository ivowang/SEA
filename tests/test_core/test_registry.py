"""Tests for the component registry."""

import pytest

from sea.core.registry import Registry


def test_register_and_build():
    reg = Registry("test")

    @reg.register("foo")
    class Foo:
        def __init__(self, x=1):
            self.x = x

    instance = reg.build("foo", x=42)
    assert instance.x == 42


def test_duplicate_register_raises():
    reg = Registry("test")

    @reg.register("bar")
    class Bar:
        pass

    with pytest.raises(KeyError, match="already registered"):
        @reg.register("bar")
        class Bar2:
            pass


def test_build_unknown_raises():
    reg = Registry("test")
    with pytest.raises(KeyError, match="not found"):
        reg.build("nonexistent")


def test_registry_repr():
    reg = Registry("test")

    @reg.register("a")
    class A:
        pass

    @reg.register("b")
    class B:
        pass

    assert "a" in repr(reg)
    assert "b" in repr(reg)
