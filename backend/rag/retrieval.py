"""RAG Retrieval Module.

Converts search queries into embeddings and queries the FAISS vector index
with similarity thresholds and session isolation.
"""

from typing import List, Dict, Any
from multimodal_ai_companion.backend.core.config import settings
from multimodal_ai_companion.backend.providers.factory import get_embedding_provider
from multimodal_ai_companion.backend.rag.vector_store import get_vector_store


def retrieve_context(
    session_id: str,
    query: str,
    top_k: int = settings.MAX_RETRIEVAL_CHUNKS,
    similarity_threshold: float = settings.SIMILARITY_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Retrieves semantically relevant document chunks strictly for the given session.

    Returns:
        List of dicts: [
            {
                "document_id": str,
                "filename": str,
                "chunk_index": int,
                "chunk_text": str,
                "score": float
            }
        ]
    """
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store(dim=embedding_provider.dimension)

    # 1. Embed query
    query_vector = embedding_provider.embed_query(query)

    # 2. Search FAISS index with session isolation
    matches = vector_store.search(
        query_vector=query_vector,
        session_id=session_id,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    return matches

