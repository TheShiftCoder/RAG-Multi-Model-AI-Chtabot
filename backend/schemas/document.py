"""Pydantic Schemas for Document Ingestion and Management."""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class DocumentInfoSchema(BaseModel):
    """Schema representing an uploaded document."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    filename: str
    file_type: str
    file_size: int
    processing_status: str
    chunk_count: int = 0
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    """Response returned upon uploading a document."""
    document_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    chunks_created: int
    message: str


class DocumentListResponse(BaseModel):
    """Response returning all documents belonging to a session."""
    session_id: str
    documents: List[DocumentInfoSchema]
