"""Embedding Provider Implementations.

Supports OpenAI, Gemini, and Local Deterministic Vector Generation
to enable zero-dependency offline RAG indexing and testing.
"""

import math
import hashlib
import numpy as np
from typing import List
from multimodal_ai_companion.backend.providers.base import BaseEmbeddingProvider
from multimodal_ai_companion.backend.core.config import settings


class LocalDeterministicEmbeddingProvider(BaseEmbeddingProvider):
    """Generates deterministic, normalized 384-dimensional embeddings locally.

    Uses character n-grams and hashing to project text into semantic vector space.
    Guarantees fast, offline, reproducible vector operations for FAISS.
    """

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_token(self, token: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        # Hash full token and 3-char ngrams
        token_clean = token.lower().strip()
        if not token_clean:
            return vec

        ngrams = [token_clean[i:i + 3] for i in range(max(1, len(token_clean) - 2))]
        ngrams.append(token_clean)

        for gram in ngrams:
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[idx] += sign

        return vec

    def _text_to_vector(self, text: str) -> List[float]:
        words = text.split()
        if not words:
            vec = np.zeros(self._dim, dtype=np.float32)
            vec[0] = 1.0
            return vec.tolist()

        vec = np.zeros(self._dim, dtype=np.float32)
        for w in words:
            vec += self._hash_token(w)

        norm = np.linalg.norm(vec)
        if norm > 1e-7:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return vec.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._text_to_vector(query)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI Embedding Provider using text-embedding-3-small."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dim = 1536

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        response = self.client.embeddings.create(input=[query], model=self.model)
        return response.data[0].embedding

