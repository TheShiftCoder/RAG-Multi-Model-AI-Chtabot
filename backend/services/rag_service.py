"""RAG Ingestion and Document Lifecycle Service."""

import os
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from fastapi import UploadFile, HTTPException

from multimodal_ai_companion.backend.core.config import settings
from multimodal_ai_companion.backend.core.security import (
    validate_file_upload,
    generate_secure_storage_path,
)
from multimodal_ai_companion.backend.models.document import Document, DocumentChunk
from multimodal_ai_companion.backend.rag.ingestion import extract_text_from_file
from multimodal_ai_companion.backend.rag.chunking import chunk_text
from multimodal_ai_companion.backend.rag.vector_store import get_vector_store
from multimodal_ai_companion.backend.rag.retrieval import retrieve_context
from multimodal_ai_companion.backend.providers.factory import get_embedding_provider

logger = logging.getLogger(__name__)


class RAGService:
    """Manages document uploads, vector embedding, and retrieval lifecycle."""

    @classmethod
    async def process_document_upload(
        cls,
        db: Session,
        session_id: str,
        file: UploadFile
    ) -> Document:
        """Validates, stores, extracts, chunks, and indexes an uploaded document."""
        # 1. Validate file extension
        ext = validate_file_upload(file, settings.ALLOWED_DOCUMENT_EXTENSIONS)

        # 2. Secure storage path
        storage_dir = Path(settings.UPLOAD_PATH)
        target_path, unique_filename = generate_secure_storage_path(storage_dir, file.filename)

        # 3. Stream write to disk while monitoring file size
        total_size = 0
        try:
            with open(target_path, "wb") as buffer:
                while content := await file.read(1024 * 1024):  # 1MB chunk
                    total_size += len(content)
                    if total_size > settings.MAX_UPLOAD_SIZE_BYTES:
                        raise HTTPException(
                            status_code=400,
                            detail=f"File exceeds maximum upload limit of {settings.MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB."
                        )
                    buffer.write(content)
        except Exception as e:
            if target_path.exists():
                os.remove(target_path)
            raise e

        # 4. Create Document record in DB
        doc_record = Document(
            id=str(uuid.uuid4()),
            session_id=session_id,
            filename=file.filename,
            file_type=ext,
            file_size=total_size,
            storage_path=str(target_path),
            processing_status="pending"
        )
        db.add(doc_record)
        db.commit()
        db.refresh(doc_record)

        # 5. Extract text
        try:
            raw_text = extract_text_from_file(target_path, ext)
            if not raw_text.strip():
                raise ValueError("The uploaded document contains no readable text.")

            # 6. Chunk text
            chunks = chunk_text(
                raw_text,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )

            if not chunks:
                raise ValueError("No text chunks could be generated from document.")

            # 7. Generate embeddings
            embedding_provider = get_embedding_provider()
            vectors = embedding_provider.embed_texts(chunks)

            # 8. Save chunks in DB and FAISS
            vector_store = get_vector_store(dim=embedding_provider.dimension)
            chunk_records: List[DocumentChunk] = []
            faiss_metadata: List[Dict[str, Any]] = []

            for idx, (chunk_str, vec) in enumerate(zip(chunks, vectors)):
                chunk_id = str(uuid.uuid4())
                chunk_rec = DocumentChunk(
                    id=chunk_id,
                    document_id=doc_record.id,
                    chunk_index=idx,
                    chunk_text=chunk_str,
                    embedding_reference=chunk_id
                )
                chunk_records.append(chunk_rec)
                faiss_metadata.append({
                    "document_id": doc_record.id,
                    "chunk_id": chunk_id,
                    "session_id": session_id,
                    "filename": doc_record.filename,
                    "chunk_index": idx,
                    "text": chunk_str
                })

            db.add_all(chunk_records)
            vector_store.add_vectors(vectors, faiss_metadata)

            doc_record.processing_status = "processed"
            db.commit()
            db.refresh(doc_record)
            logger.info(f"Successfully processed document {doc_record.filename} with {len(chunks)} chunks.")
            return doc_record

        except Exception as e:
            logger.error(f"Failed to process document {doc_record.id}: {e}")
            doc_record.processing_status = "failed"
            db.commit()
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

    @classmethod
    def get_session_documents(cls, db: Session, session_id: str) -> List[Document]:
        """Returns all documents uploaded in a specific session."""
        return db.query(Document).filter(Document.session_id == session_id).all()

    @classmethod
    def get_document_by_id(cls, db: Session, document_id: str) -> Optional[Document]:
        """Fetches document by its UUID."""
        return db.query(Document).filter(Document.id == document_id).first()

    @classmethod
    def delete_document(cls, db: Session, document_id: str) -> bool:
        """Deletes a document from DB, removes its vector embeddings from FAISS, and removes disk file."""
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return False

        # 1. Remove vectors from FAISS
        embedding_provider = get_embedding_provider()
        vector_store = get_vector_store(dim=embedding_provider.dimension)
        vector_store.delete_document_vectors(document_id)

        # 2. Remove physical file
        if doc.storage_path and os.path.exists(doc.storage_path):
            try:
                os.remove(doc.storage_path)
            except Exception as e:
                logger.warning(f"Failed to delete file from disk: {e}")

        # 3. Delete from DB (cascades to chunks)
        db.delete(doc)
        db.commit()
        return True

    @classmethod
    def retrieve_session_context(
        cls,
        session_id: str,
        query: str,
        top_k: int = settings.MAX_RETRIEVAL_CHUNKS,
        similarity_threshold: float = settings.SIMILARITY_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """Retrieves semantically relevant document chunks strictly for the given session."""
        return retrieve_context(
            session_id=session_id,
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

