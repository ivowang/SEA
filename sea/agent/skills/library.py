"""Skill library: file-system backed collection with progressive disclosure.

Skills are stored as SKILL.md files and indexed for retrieval.
Supports both embedding-based (FAISS) and keyword-based retrieval.
FAISS/sentence-transformers are optional — keyword fallback works without them.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from sea.agent.skills.base import Skill
from sea.agent.skills.disclosure import (
    DisclosureLevel,
    SkillView,
    skill_to_view,
)
from sea.agent.skills.skill_md import (
    SkillFrontmatter,
    SkillMd,
    _sanitize_filename,
    load_skill_md,
    save_skill_md,
    skill_from_dict,
    skill_to_dict,
)
from sea.core.base import Evolvable
from sea.core.registry import SKILL_REGISTRY

logger = logging.getLogger(__name__)


@SKILL_REGISTRY.register("default")
class SkillLibrary(Evolvable[list[dict[str, Any]]]):
    """Manages skills with file-system storage and progressive disclosure.

    Skills are stored as SkillMd objects (SKILL.md format) in memory,
    optionally backed by a directory of .md files on disk.

    Retrieval supports two modes:
    - Embedding-based (FAISS + SentenceTransformer) — opt-in, requires extra deps
    - Keyword-based (Jaccard overlap on name/description/tags) — default fallback
    """

    def __init__(
        self,
        skills_dir: Path | str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        use_embeddings: bool = True,
        default_disclosure: DisclosureLevel = DisclosureLevel.SUMMARY,
    ) -> None:
        self._skills_dir = Path(skills_dir) if skills_dir else None
        self._embedding_model_name = embedding_model
        self._embedding_dim = embedding_dim
        self._use_embeddings = use_embeddings
        self._default_disclosure = default_disclosure

        self._skills: dict[str, SkillMd] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._embedder = None
        self._index = None

        # Load from directory if provided
        if self._skills_dir and self._skills_dir.exists():
            self._load_from_dir()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add_skill(self, skill: Skill | SkillMd | dict) -> None:
        """Add or update a skill. Accepts SkillMd, legacy Skill, or dict."""
        md = self._to_skill_md(skill)
        self._skills[md.name] = md

        # Write to disk if skills_dir is set
        if self._skills_dir:
            self._skills_dir.mkdir(parents=True, exist_ok=True)
            filename = _sanitize_filename(md.name) + ".md"
            save_skill_md(md, self._skills_dir / filename)

        # Update embedding index
        if self._use_embeddings:
            try:
                self._ensure_embedder()
                self._embeddings[md.name] = self._embed(
                    f"{md.name} {md.description} {' '.join(md.frontmatter.tags)}"
                )
                self._rebuild_index()
            except ImportError:
                pass  # embeddings not available, use keyword fallback

        logger.info("Added skill '%s' (%d total)", md.name, len(self._skills))

    def remove_skill(self, name: str) -> None:
        """Remove a skill by name."""
        if name in self._skills:
            md = self._skills.pop(name)
            self._embeddings.pop(name, None)

            # Remove file from disk
            if md.file_path and md.file_path.exists():
                md.file_path.unlink()
            elif self._skills_dir:
                fp = self._skills_dir / (_sanitize_filename(name) + ".md")
                if fp.exists():
                    fp.unlink()

            if self._index is not None:
                self._rebuild_index()

    def get_skill(self, name: str) -> SkillMd | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[SkillMd]:
        """Return all skills."""
        return list(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    # ------------------------------------------------------------------
    # Progressive Disclosure Retrieval
    # ------------------------------------------------------------------

    def get_index(self) -> list[SkillView]:
        """Return all skills at INDEX level (name + description only).

        This is lightweight and suitable for always-in-context display.
        """
        return [skill_to_view(s, DisclosureLevel.INDEX) for s in self._skills.values()]

    def retrieve(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.3,
        level: DisclosureLevel | None = None,
    ) -> list[SkillView]:
        """Retrieve the most relevant skills at the specified disclosure level.

        Args:
            query: Search query text.
            k: Maximum number of results.
            threshold: Minimum relevance score.
            level: Disclosure level (defaults to self._default_disclosure).

        Returns:
            List of SkillView objects at the requested disclosure level.
        """
        if not self._skills:
            return []

        level = level or self._default_disclosure
        k = min(k, len(self._skills))

        # Try embedding-based retrieval, fall back to keyword
        if self._use_embeddings and self._embedder is not None and self._index is not None:
            matches = self._embedding_retrieve(query, k, threshold)
        else:
            matches = self._keyword_retrieve(query, k, threshold)

        return [skill_to_view(self._skills[name], level) for name, _score in matches]

    def retrieve_full(self, name: str) -> SkillView | None:
        """Retrieve full content of a specific skill by name.

        Used for on-demand progressive disclosure (read_skill tool).
        """
        skill = self._skills.get(name)
        if skill is None:
            return None
        return skill_to_view(skill, DisclosureLevel.FULL)

    # ------------------------------------------------------------------
    # Embedding-based retrieval (optional, requires FAISS)
    # ------------------------------------------------------------------

    def _ensure_embedder(self) -> None:
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer
        import faiss  # noqa: F401 — validate availability
        self._embedder = SentenceTransformer(self._embedding_model_name)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        import faiss
        import numpy as np
        self._index = faiss.IndexFlatIP(self._embedding_dim)
        if self._embeddings:
            vecs = np.array(list(self._embeddings.values()), dtype=np.float32)
            self._index.add(vecs)

    def _embed(self, text: str) -> list[float]:
        emb = self._embedder.encode([text], normalize_embeddings=True)
        return emb[0].tolist()

    def _embedding_retrieve(
        self, query: str, k: int, threshold: float,
    ) -> list[tuple[str, float]]:
        import numpy as np
        query_emb = np.array([self._embed(query)], dtype=np.float32)
        scores, indices = self._index.search(query_emb, k)
        names = list(self._skills.keys())
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < threshold:
                continue
            if 0 <= idx < len(names):
                results.append((names[idx], float(score)))
        return results

    # ------------------------------------------------------------------
    # Keyword-based retrieval (zero external deps)
    # ------------------------------------------------------------------

    def _keyword_retrieve(
        self, query: str, k: int, threshold: float,
    ) -> list[tuple[str, float]]:
        """Retrieve skills by keyword overlap (Jaccard on tokens)."""
        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens:
            return []

        scored: list[tuple[str, float]] = []
        for name, skill in self._skills.items():
            # Build skill token set from name + description + tags
            skill_text = f"{name} {skill.description} {' '.join(skill.frontmatter.tags)}"
            skill_tokens = set(re.findall(r"\w+", skill_text.lower()))
            if not skill_tokens:
                continue

            # Jaccard similarity
            intersection = len(query_tokens & skill_tokens)
            union = len(query_tokens | skill_tokens)
            score = intersection / union if union > 0 else 0.0

            if score >= threshold:
                scored.append((name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------
    # File system operations
    # ------------------------------------------------------------------

    def _load_from_dir(self) -> None:
        """Load all .md files from skills_dir."""
        if not self._skills_dir or not self._skills_dir.exists():
            return
        for md_path in sorted(self._skills_dir.glob("*.md")):
            try:
                skill = load_skill_md(md_path)
                self._skills[skill.name] = skill
            except Exception as e:
                logger.warning("Failed to load skill from %s: %s", md_path, e)

        if self._skills:
            logger.info("Loaded %d skills from %s", len(self._skills), self._skills_dir)

    # ------------------------------------------------------------------
    # Checkpointable
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        # Write individual .md files
        md_dir = path / "md"
        md_dir.mkdir(exist_ok=True)
        for skill in self._skills.values():
            filename = _sanitize_filename(skill.name) + ".md"
            save_skill_md(skill, md_dir / filename)

        # Also write legacy JSON for backward compat
        legacy_data = [skill_to_dict(s) for s in self._skills.values()]
        (path / "skill_library.json").write_text(
            json.dumps(legacy_data, ensure_ascii=False, indent=2)
        )

        # Write lightweight index
        index_data = [
            {"name": s.name, "description": s.description, "tags": s.frontmatter.tags}
            for s in self._skills.values()
        ]
        (path / "skill_index.json").write_text(
            json.dumps(index_data, ensure_ascii=False, indent=2)
        )

    def load_checkpoint(self, path: Path) -> None:
        self._skills.clear()
        self._embeddings.clear()

        # Try loading from .md files first, fall back to JSON if no skills loaded
        md_dir = path / "md"
        if md_dir.exists():
            for md_path in sorted(md_dir.glob("*.md")):
                try:
                    skill = load_skill_md(md_path)
                    self._skills[skill.name] = skill
                except Exception as e:
                    logger.warning("Failed to load skill from %s: %s", md_path, e)

        if not self._skills and (path / "skill_library.json").exists():
            # Fallback to legacy JSON
            data = json.loads((path / "skill_library.json").read_text())
            for d in data:
                md = skill_from_dict(d)
                self._skills[md.name] = md
            logger.info("Migrated %d skills from legacy JSON format", len(self._skills))

        # Rebuild embeddings if available
        self._embedder = None
        self._index = None
        if self._skills and self._use_embeddings:
            try:
                self._ensure_embedder()
                for name, skill in self._skills.items():
                    text = f"{name} {skill.description} {' '.join(skill.frontmatter.tags)}"
                    self._embeddings[name] = self._embed(text)
                self._rebuild_index()
            except ImportError:
                logger.info("FAISS/sentence-transformers not available, using keyword retrieval")

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_skills": len(self._skills),
            "skill_names": list(self._skills.keys()),
        }

    # ------------------------------------------------------------------
    # Evolvable
    # ------------------------------------------------------------------

    def get_evolvable_state(self) -> list[dict[str, Any]]:
        return [skill_to_dict(s) for s in self._skills.values()]

    def set_evolvable_state(self, state: list[dict[str, Any]]) -> None:
        self._skills.clear()
        self._embeddings.clear()
        for d in state:
            md = skill_from_dict(d)
            self._skills[md.name] = md

        # Reconcile disk: remove stale files, write new ones
        if self._skills_dir and self._skills_dir.exists():
            for old_md in self._skills_dir.glob("*.md"):
                old_md.unlink()
        if self._skills_dir:
            self._skills_dir.mkdir(parents=True, exist_ok=True)
            for skill in self._skills.values():
                filename = _sanitize_filename(skill.name) + ".md"
                save_skill_md(skill, self._skills_dir / filename)

        # Rebuild embeddings
        self._embedder = None
        self._index = None
        if self._skills and self._use_embeddings:
            try:
                self._ensure_embedder()
                for name, skill in self._skills.items():
                    text = f"{name} {skill.description} {' '.join(skill.frontmatter.tags)}"
                    self._embeddings[name] = self._embed(text)
                self._rebuild_index()
            except ImportError:
                pass

        logger.info("Updated skill library: %d skills", len(self._skills))

    def evolution_metadata(self) -> dict[str, Any]:
        return {
            "type": "skill_library",
            "num_skills": len(self._skills),
            "skill_names": list(self._skills.keys()),
            "storage": "filesystem" if self._skills_dir else "memory",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_skill_md(skill: Skill | SkillMd | dict) -> SkillMd:
        """Convert any skill representation to SkillMd."""
        if isinstance(skill, SkillMd):
            return skill
        if isinstance(skill, dict):
            return skill_from_dict(skill)
        if isinstance(skill, Skill):
            # Convert legacy Skill object
            if hasattr(skill, "to_skill_md"):
                return skill.to_skill_md()
            # Generic conversion
            fm = SkillFrontmatter(
                name=skill.info.name,
                description=skill.info.description,
                tags=skill.info.tags,
                sub_skills=skill.info.sub_skills,
            )
            return SkillMd(frontmatter=fm, body=skill.to_prompt())
        raise TypeError(f"Cannot convert {type(skill).__name__} to SkillMd")
