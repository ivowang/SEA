"""Semantic memory: FAISS-backed vector store with embedding retrieval.

This memory is Evolvable — its contents can be curated, compressed, or
reorganised by evolution methods.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sea.agent.memory.base import Memory, MemoryEntry
from sea.core.base import Evolvable
from sea.core.registry import MEMORY_REGISTRY

logger = logging.getLogger(__name__)


@MEMORY_REGISTRY.register("semantic")
class SemanticMemory(Memory, Evolvable[list[dict[str, Any]]]):
    """Long-term memory with FAISS-backed embedding retrieval.

    Uses sentence-transformers for encoding and FAISS for fast nearest-neighbour
    search.  The evolvable state is the list of entry dicts, allowing evolvers
    to add, remove, or modify memories.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        max_size: int = 5000,
    ) -> None:
        self._embedding_model_name = embedding_model
        self._embedding_dim = embedding_dim
        self._max_size = max_size
        self._entries: list[MemoryEntry] = []
        self._embedder = None
        self._index = None

    def _ensure_loaded(self) -> None:
        """Lazy-load embedding model and FAISS index."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                try:
                    import faiss
                except ImportError:
                    raise ImportError(
                        "SemanticMemory requires sentence-transformers and faiss. "
                        "Install with: pip install sentence-transformers faiss-cpu"
                    )
            except ImportError as e:
                raise ImportError(
                    "SemanticMemory requires sentence-transformers and faiss. "
                    "Install with: pip install sentence-transformers faiss-cpu"
                ) from e
            self._embedder = SentenceTransformer(self._embedding_model_name)
            self._index = faiss.IndexFlatIP(self._embedding_dim)
            logger.info("Loaded embedding model: %s", self._embedding_model_name)

    def _embed(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        embeddings = self._embedder.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from current entries."""
        self._ensure_loaded()
        import faiss

        self._index = faiss.IndexFlatIP(self._embedding_dim)
        if self._entries:
            embeddings = []
            for entry in self._entries:
                if entry.embedding is not None:
                    embeddings.append(entry.embedding)
                else:
                    emb = self._embed([entry.content])[0]
                    entry.embedding = emb.tolist()
                    embeddings.append(emb.tolist())
            self._index.add(np.array(embeddings, dtype=np.float32))

    def add(self, entry: MemoryEntry) -> None:
        self._ensure_loaded()
        if entry.embedding is None:
            emb = self._embed([entry.content])[0]
            entry.embedding = emb.tolist()
        self._entries.append(entry)
        self._index.add(np.array([entry.embedding], dtype=np.float32))

        if len(self._entries) > self._max_size:
            self._entries = self._entries[-self._max_size :]
            self._rebuild_index()

    def retrieve(self, query: str, k: int = 5) -> list[MemoryEntry]:
        self._ensure_loaded()
        if not self._entries:
            return []
        k = min(k, len(self._entries))
        query_emb = self._embed([query])
        scores, indices = self._index.search(query_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._entries):
                continue
            entry = self._entries[idx]
            entry.score = float(score)
            results.append(entry)
        return results

    def get_all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        if self._index is not None:
            self._index.reset()

    # -- Checkpointable --

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        data = []
        for e in self._entries:
            d = e.to_dict()
            d["embedding"] = e.embedding
            data.append(d)
        (path / "semantic_memory.json").write_text(json.dumps(data, ensure_ascii=False))

    def load_checkpoint(self, path: Path) -> None:
        fp = path / "semantic_memory.json"
        if fp.exists():
            data = json.loads(fp.read_text())
            self._entries = [
                MemoryEntry(
                    content=d["content"],
                    embedding=d.get("embedding"),
                    metadata=d.get("metadata", {}),
                    timestamp=d.get("timestamp", 0.0),
                    memory_type=d.get("memory_type", "semantic"),
                )
                for d in data
            ]
            self._rebuild_index()

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_entries": len(self._entries),
            "embedding_model": self._embedding_model_name,
            "max_size": self._max_size,
        }

    # -- Evolvable --

    def get_evolvable_state(self) -> list[dict[str, Any]]:
        result = []
        for e in self._entries:
            d = e.to_dict()
            d["embedding"] = e.embedding
            result.append(d)
        return result

    def set_evolvable_state(self, state: list[dict[str, Any]]) -> None:
        self._entries = [
            MemoryEntry(
                content=d["content"],
                embedding=d.get("embedding"),
                metadata=d.get("metadata", {}),
                timestamp=d.get("timestamp", 0.0),
                memory_type=d.get("memory_type", "semantic"),
            )
            for d in state
        ]
        self._rebuild_index()
        logger.info("Updated semantic memory: %d entries", len(self._entries))

    def evolution_metadata(self) -> dict[str, Any]:
        return {
            "type": "semantic_memory",
            "num_entries": len(self._entries),
            "max_size": self._max_size,
            "embedding_model": self._embedding_model_name,
        }
