"""Tests for memory implementations."""

import tempfile
from pathlib import Path

from sea.agent.memory.base import MemoryEntry
from sea.agent.memory.working import WorkingMemory
from sea.agent.memory.episodic import EpisodicMemory


def test_working_memory_add_retrieve():
    mem = WorkingMemory(max_size=5)
    for i in range(3):
        mem.add(MemoryEntry(content=f"entry_{i}"))

    results = mem.retrieve("anything", k=2)
    assert len(results) == 2
    assert results[-1].content == "entry_2"  # Most recent


def test_working_memory_overflow():
    mem = WorkingMemory(max_size=3)
    for i in range(5):
        mem.add(MemoryEntry(content=f"entry_{i}"))

    assert mem.size() == 3
    all_entries = mem.get_all()
    assert all_entries[0].content == "entry_2"  # Oldest remaining


def test_working_memory_checkpoint():
    mem = WorkingMemory(max_size=10)
    mem.add(MemoryEntry(content="hello"))
    mem.add(MemoryEntry(content="world"))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "mem"
        mem.save_checkpoint(path)

        mem2 = WorkingMemory(max_size=10)
        mem2.load_checkpoint(path)
        assert mem2.size() == 2
        assert mem2.get_all()[0].content == "hello"


def test_episodic_memory_retrieve_by_keyword():
    mem = EpisodicMemory(max_size=100)
    mem.add(MemoryEntry(content="The cat sat on the mat"))
    mem.add(MemoryEntry(content="The dog ran in the park"))
    mem.add(MemoryEntry(content="A cat played with yarn"))

    results = mem.retrieve("cat", k=2)
    assert len(results) == 2
    # Both cat entries should score higher
    assert all("cat" in r.content.lower() for r in results)


def test_episodic_memory_clear():
    mem = EpisodicMemory()
    mem.add(MemoryEntry(content="test"))
    assert mem.size() == 1
    mem.clear()
    assert mem.size() == 0
