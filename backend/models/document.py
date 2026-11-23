"""SQLAlchemy Model for Documents and Document Chunks."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from multimodal_ai_companion.backend.database.session import Base


class Document(Base):
    """Document model representing an uploaded file."""

    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(64), index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    storage_path = Column(String(500), nullable=False)
    processing_status = Column(
        String(50), nullable=False, default="pending"
    )  # 'pending', 'processed', 'failed'
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename='{self.filename}' status={self.processing_status}>"


class DocumentChunk(Base):
    """DocumentChunk model storing chunked text and vector references."""

    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding_reference = Column(String(100), nullable=True)  # FAISS vector id/index
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    document = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<DocumentChunk id={self.id} doc={self.document_id} idx={self.chunk_index}>"

