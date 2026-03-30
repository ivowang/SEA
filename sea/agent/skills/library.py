"""Skill library: indexed collection of skills with retrieval.

The SkillLibrary is Evolvable — evolution methods can add, remove,
or refine skills.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sea.agent.skills.base import Skill, SkillInfo
from sea.agent.skills.code_skill import CodeSkill, TextSkill
from sea.core.base import Evolvable
from sea.core.registry import SKILL_REGISTRY

logger = logging.getLogger(__name__)


@SKILL_REGISTRY.register("default")
class SkillLibrary(Evolvable[list[dict[str, Any]]]):
    """Manages a collection of skills with embedding-based retrieval.

    Skills are indexed by their description embeddings for semantic search.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
    ) -> None:
        self._embedding_model_name = embedding_model
        self._embedding_dim = embedding_dim
        self._skills: dict[str, Skill] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._embedder = None
        self._index = None

    def _ensure_loaded(self) -> None:
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                import faiss
            except ImportError as e:
                raise ImportError(
                    "SkillLibrary requires sentence-transformers and faiss. "
                    "Install with: pip install sentence-transformers faiss-cpu"
                ) from e
            self._embedder = SentenceTransformer(self._embedding_model_name)
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        import faiss

        self._index = faiss.IndexFlatIP(self._embedding_dim)
        if self._embeddings:
            vecs = np.array(list(self._embeddings.values()), dtype=np.float32)
            self._index.add(vecs)

    def _embed(self, text: str) -> list[float]:
        self._ensure_loaded()
        emb = self._embedder.encode([text], normalize_embeddings=True)
        return emb[0].tolist()

    def add_skill(self, skill: Skill) -> None:
        """Add or update a skill in the library."""
        self._ensure_loaded()
        self._skills[skill.name] = skill
        self._embeddings[skill.name] = self._embed(skill.description)
        self._rebuild_index()
        logger.info("Added skill '%s' (%d total)", skill.name, len(self._skills))

    def remove_skill(self, name: str) -> None:
        """Remove a skill by name."""
        if name in self._skills:
            del self._skills[name]
            del self._embeddings[name]
            self._rebuild_index()

    def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def retrieve(self, query: str, k: int = 5, threshold: float = 0.3) -> list[Skill]:
        """Retrieve the most relevant skills for *query*.

        Only returns skills with cosine similarity >= threshold.
        Returns empty list if no skill is relevant enough.
        """
        self._ensure_loaded()
        if not self._skills:
            return []
        k = min(k, len(self._skills))
        query_emb = np.array([self._embed(query)], dtype=np.float32)
        scores, indices = self._index.search(query_emb, k)
        names = list(self._skills.keys())
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < threshold:
                continue  # skip irrelevant skills
            if 0 <= idx < len(names):
                results.append(self._skills[names[idx]])
        return results

    def list_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    # -- Checkpointable --

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        data = [s.to_dict() for s in self._skills.values()]
        (path / "skill_library.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load_checkpoint(self, path: Path) -> None:
        fp = path / "skill_library.json"
        if not fp.exists():
            return
        data = json.loads(fp.read_text())
        self._skills.clear()
        self._embeddings.clear()
        for d in data:
            skill = self._dict_to_skill(d)
            self._skills[skill.name] = skill
        # Rebuild embeddings immediately
        self._embedder = None
        self._index = None
        self._embeddings.clear()
        if self._skills:
            self._ensure_loaded()
            for name, skill in self._skills.items():
                self._embeddings[name] = self._embed(skill.description)
            self._rebuild_index()

    def state_dict(self) -> dict[str, Any]:
        return {"num_skills": len(self._skills), "skill_names": list(self._skills.keys())}

    # -- Evolvable --

    def get_evolvable_state(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._skills.values()]

    def set_evolvable_state(self, state: list[dict[str, Any]]) -> None:
        self._skills.clear()
        self._embeddings.clear()
        for d in state:
            skill = self._dict_to_skill(d)
            self._skills[skill.name] = skill
        # Rebuild embeddings and FAISS index (same as load_checkpoint)
        self._embedder = None
        self._index = None
        if self._skills:
            self._ensure_loaded()
            for name, skill in self._skills.items():
                self._embeddings[name] = self._embed(skill.description)
            self._rebuild_index()
        logger.info("Updated skill library: %d skills", len(self._skills))

    def evolution_metadata(self) -> dict[str, Any]:
        return {
            "type": "skill_library",
            "num_skills": len(self._skills),
            "skill_names": list(self._skills.keys()),
        }

    @staticmethod
    def _dict_to_skill(d: dict[str, Any]) -> Skill:
        if "source_code" in d:
            return CodeSkill.from_dict(d)
        elif "sub_skills" in d or "composition_plan" in d:
            from sea.agent.skills.code_skill import ComposedSkill
            return ComposedSkill.from_dict(d)
        elif "instructions" in d:
            return TextSkill.from_dict(d)
        else:
            return TextSkill(
                name=d["name"],
                description=d.get("description", ""),
                instructions=d.get("description", ""),
                tags=d.get("tags", []),
            )
