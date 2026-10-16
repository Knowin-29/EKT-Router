"""Knowledge module: template retrieval + optional scene RAG.

Design goals
------------
- Zero-cost fallback when weights / indices are missing.
- Unified string output for downstream prompt concatenation.
- Consistent interface with :class:`modules.tool_module.ToolModule`.

Modes
-----
- ``none`` returns ``""``.
- ``template`` performs keyword matching over ``knowledge/template_db.json``.
- ``rag`` performs nearest-neighbor retrieval on a FAISS (or NumPy cosine)
  index stored under ``knowledge/scene_rag/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class KnowledgeModule:
    def __init__(
        self,
        template_db_path: str,
        rag_index_dir: Optional[str] = None,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        top_n: int = 3,
        dry_run: bool = False,
    ) -> None:
        self.template_db_path = Path(template_db_path)
        self.rag_index_dir = Path(rag_index_dir) if rag_index_dir else None
        self.embedding_model_name = embedding_model_name
        self.top_n = top_n
        self.dry_run = dry_run

        self.template_db: dict = {}
        if self.template_db_path.exists():
            try:
                self.template_db = json.loads(self.template_db_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("failed to load template_db %s: %s", self.template_db_path, e)

        self._rag_ready = False
        self._rag_embedder = None
        self._rag_index = None
        self._rag_passages: List[str] = []

    def __call__(self, question: str, mode: str) -> str:
        mode = (mode or "none").lower()
        if self.dry_run:
            return f"[knowledge:{mode}:dry-run]" if mode != "none" else ""
        try:
            if mode == "template":
                return self._retrieve_template(question)
            if mode == "rag":
                return self._retrieve_scene(question)
        except Exception as e:
            logger.warning("knowledge module failure (mode=%s): %s", mode, e)
        return ""

    def _retrieve_template(self, question: str) -> str:
        q = (question or "").lower()
        for item in self.template_db.values():
            kws = [k.lower() for k in item.get("keywords", [])]
            if any(k in q for k in kws):
                return item.get("template", "")
        # Fallback: deterministic first template to avoid empty prompt noise
        if self.template_db:
            first = next(iter(self.template_db.values()))
            return first.get("template", "")
        return ""

    def _ensure_rag(self) -> bool:
        if self._rag_ready:
            return True
        if self.rag_index_dir is None or not self.rag_index_dir.exists():
            return False
        passages_file = self.rag_index_dir / "passages.jsonl"
        index_file = self.rag_index_dir / "index.faiss"
        metadata_file = self.rag_index_dir / "metadata.json"
        try:
            if passages_file.exists():
                with passages_file.open("r", encoding="utf-8") as f:
                    self._rag_passages = [json.loads(line).get("text", "") for line in f if line.strip()]
            elif metadata_file.exists():
                meta = json.loads(metadata_file.read_text(encoding="utf-8"))
                self._rag_passages = [m.get("text", "") for m in meta]
            if not self._rag_passages:
                return False
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._rag_embedder = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                logger.warning("sentence-transformers unavailable: %s", e)
                return False
            if index_file.exists():
                try:
                    import faiss  # type: ignore
                    self._rag_index = faiss.read_index(str(index_file))
                except Exception as e:
                    logger.warning("faiss unavailable: %s", e)
                    self._rag_index = None
            self._rag_ready = True
            return True
        except Exception as e:
            logger.warning("failed to load RAG index: %s", e)
            return False

    def _retrieve_scene(self, question: str) -> str:
        if not self._ensure_rag():
            return ""
        import numpy as np  # local import

        q_vec = self._rag_embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        if self._rag_index is not None:
            _, idx = self._rag_index.search(q_vec.astype("float32"), self.top_n)
            idx = idx[0].tolist()
        else:
            # fallback: numpy cosine over encoded passages (lazy)
            if not hasattr(self, "_rag_passage_vecs"):
                vecs = self._rag_embedder.encode(
                    self._rag_passages, convert_to_numpy=True, normalize_embeddings=True
                )
                self._rag_passage_vecs = vecs.astype("float32")
            sims = self._rag_passage_vecs @ q_vec[0].astype("float32")
            idx = np.argsort(-sims)[: self.top_n].tolist()
        snippets = [self._rag_passages[i] for i in idx if 0 <= i < len(self._rag_passages)]
        return "\n---\n".join(snippets)
