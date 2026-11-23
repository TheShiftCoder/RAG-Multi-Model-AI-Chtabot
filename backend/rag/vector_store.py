"""FAISS and Resilient Vector Storage Module.

Implements cosine similarity search with FAISS IndexFlatIP.
Includes a transparent NumPy cosine fallback when FAISS C++ OpenMP runtime DLLs
are unavailable on specific operating system builds.
Stores vector mappings with session isolation to guarantee no cross-session leakage.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from multimodal_ai_companion.backend.core.config import settings

logger = logging.getLogger(__name__)

# Attempt to load FAISS; if C++ DLL initialization fails, fall back to pure NumPy FlatIP
HAS_FAISS = False
try:
    import faiss  # type: ignore
    # Test index creation to ensure swigfaiss DLL is functional
    test_idx = faiss.IndexFlatIP(4)
    HAS_FAISS = True
    logger.info("FAISS vector engine initialized successfully.")
except Exception as e:
    logger.warning(f"FAISS native library unavailable ({e}). Operating in pure NumPy IndexFlatIP mode.")


class FAISSVectorStore:
    """Session-isolated Vector Store backed by FAISS or NumPy FlatIP."""

    def __init__(self, dimension: int = 384, storage_dir: Optional[str] = None):
        self.dimension = dimension
        self.storage_dir = Path(storage_dir or settings.VECTOR_INDEX_PATH)
        self.index_file = self.storage_dir / "faiss.index"
        self.metadata_file = self.storage_dir / "metadata.json"
        self.vectors_file = self.storage_dir / "vectors.npy"

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: List[Dict[str, Any]] = []

        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.vectors: np.ndarray = np.empty((0, self.dimension), dtype=np.float32)

        self._load_if_exists()

    def _load_if_exists(self):
        """Loads index and metadata from disk if available."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

                if HAS_FAISS and self.index_file.exists():
                    self.index = faiss.read_index(str(self.index_file))
                    logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")
                elif self.vectors_file.exists():
                    self.vectors = np.load(str(self.vectors_file))
                    logger.info(f"Loaded NumPy vector store with {len(self.vectors)} vectors.")
            except Exception as e:
                logger.warning(f"Could not load existing vector index ({e}). Initializing empty index.")
                self.metadata = []
                if HAS_FAISS:
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    self.vectors = np.empty((0, self.dimension), dtype=np.float32)

    def _save_to_disk(self):
        """Persists the vector index and metadata to disk."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)

            if HAS_FAISS:
                faiss.write_index(self.index, str(self.index_file))
            else:
                np.save(str(self.vectors_file), self.vectors)
        except Exception as e:
            logger.error(f"Failed to persist vector index to disk: {e}")

    def add_vectors(self, vectors: List[List[float]], meta_list: List[Dict[str, Any]]):
        """Adds normalized vector embeddings and their corresponding metadata."""
        if not vectors:
            return

        vec_np = np.array(vectors, dtype=np.float32)
        # L2 Normalize
        norms = np.linalg.norm(vec_np, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vec_np = vec_np / norms

        if HAS_FAISS:
            self.index.add(vec_np)
        else:
            if len(self.vectors) == 0:
                self.vectors = vec_np
            else:
                self.vectors = np.vstack([self.vectors, vec_np])

        self.metadata.extend(meta_list)
        self._save_to_disk()

    @property
    def total_vectors(self) -> int:
        if HAS_FAISS:
            return self.index.ntotal
        return len(self.vectors)

    def search(
        self,
        query_vector: List[float],
        session_id: str,
        top_k: int = 4,
        similarity_threshold: float = 0.55
    ) -> List[Dict[str, Any]]:
        """Searches for top-K vectors matching the query vector and strictly belonging to session_id."""
        if self.total_vectors == 0 or not self.metadata:
            return []

        q_np = np.array([query_vector], dtype=np.float32)
        q_norm = np.linalg.norm(q_np)
        if q_norm > 0:
            q_np = q_np / q_norm

        if HAS_FAISS:
            search_k = min(self.index.ntotal, max(top_k * 5, 20))
            distances, indices = self.index.search(q_np, search_k)
            dists = distances[0]
            idxs = indices[0]
        else:
            # Exact Flat Inner Product with NumPy
            dot_products = np.dot(self.vectors, q_np.T).flatten()
            search_k = min(len(dot_products), max(top_k * 5, 20))
            idxs = np.argsort(-dot_products)[:search_k]
            dists = dot_products[idxs]

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(dists, idxs):
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]
            # Session isolation guard
            if meta.get("session_id") != session_id:
                continue

            score = float(dist)
            if score >= similarity_threshold:
                result_entry = dict(meta)
                result_entry["score"] = score
                results.append(result_entry)
                if len(results) >= top_k:
                    break

        return results

    def delete_document_vectors(self, document_id: str):
        """Removes all vectors belonging to a specific document and rebuilds the index."""
        if self.total_vectors == 0:
            return

        retained_indices = [
            i for i, m in enumerate(self.metadata) if m.get("document_id") != document_id
        ]

        if len(retained_indices) == len(self.metadata):
            return

        if not retained_indices:
            self.metadata = []
            if HAS_FAISS:
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.vectors = np.empty((0, self.dimension), dtype=np.float32)
        else:
            if HAS_FAISS:
                retained_vectors = np.vstack([self.index.reconstruct(i) for i in retained_indices])
                new_index = faiss.IndexFlatIP(self.dimension)
                new_index.add(retained_vectors)
                self.index = new_index
            else:
                self.vectors = self.vectors[retained_indices]

            self.metadata = [self.metadata[i] for i in retained_indices]

        self._save_to_disk()

    def delete_session_vectors(self, session_id: str):
        """Removes all vectors associated with a session."""
        if self.total_vectors == 0:
            return

        retained_indices = [
            i for i, m in enumerate(self.metadata) if m.get("session_id") != session_id
        ]

        if len(retained_indices) == len(self.metadata):
            return

        if not retained_indices:
            self.metadata = []
            if HAS_FAISS:
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.vectors = np.empty((0, self.dimension), dtype=np.float32)
        else:
            if HAS_FAISS:
                retained_vectors = np.vstack([self.index.reconstruct(i) for i in retained_indices])
                new_index = faiss.IndexFlatIP(self.dimension)
                new_index.add(retained_vectors)
                self.index = new_index
            else:
                self.vectors = self.vectors[retained_indices]

            self.metadata = [self.metadata[i] for i in retained_indices]

        self._save_to_disk()


# Singleton vector store instance
_vector_store_instance: Optional[FAISSVectorStore] = None


def get_vector_store(dim: int = 384) -> FAISSVectorStore:
    global _vector_store_instance
    if _vector_store_instance is None or _vector_store_instance.dimension != dim:
        _vector_store_instance = FAISSVectorStore(dimension=dim)
    return _vector_store_instance
