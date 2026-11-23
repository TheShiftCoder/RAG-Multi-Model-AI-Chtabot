"""Document Management API Endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session

from multimodal_ai_companion.backend.database.session import get_db
from multimodal_ai_companion.backend.models.document import Document
from multimodal_ai_companion.backend.schemas.document import (
    DocumentUploadResponse,
    DocumentInfoSchema,
    DocumentListResponse,
)
from multimodal_ai_companion.backend.services.rag_service import RAGService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """Uploads a PDF or TXT document, extracts text, chunks it, and indexes into FAISS."""
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be blank.")

    doc = await RAGService.process_document_upload(
        db=db,
        session_id=session_id.strip(),
        file=file
    )

    return DocumentUploadResponse(
        document_id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        file_size=doc.file_size,
        status=doc.processing_status,
        chunks_created=len(doc.chunks),
        message="Document uploaded and indexed successfully into FAISS."
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    session_id: str = Query(..., description="Unique session identifier"),
    db: Session = Depends(get_db)
):
    """Lists all documents uploaded within the session."""
    docs = RAGService.get_session_documents(db=db, session_id=session_id)
    doc_schemas = [
        DocumentInfoSchema(
            id=d.id,
            session_id=d.session_id,
            filename=d.filename,
            file_type=d.file_type,
            file_size=d.file_size,
            processing_status=d.processing_status,
            chunk_count=len(d.chunks),
            created_at=d.created_at,
        )
        for d in docs
    ]
    return DocumentListResponse(session_id=session_id, documents=doc_schemas)


@router.get("/{document_id}", response_model=DocumentInfoSchema)
def get_document(
    document_id: str,
    session_id: str = Query(..., description="Session identifier"),
    db: Session = Depends(get_db)
):
    """Fetches metadata for a specific document."""
    doc = RAGService.get_document_by_id(db=db, document_id=document_id)
    if not doc or doc.session_id != session_id:
        raise HTTPException(status_code=404, detail="Document not found.")

    return DocumentInfoSchema(
        id=doc.id,
        session_id=doc.session_id,
        filename=doc.filename,
        file_type=doc.file_type,
        file_size=doc.file_size,
        processing_status=doc.processing_status,
        chunk_count=len(doc.chunks),
        created_at=doc.created_at,
    )


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    session_id: str = Query(..., description="Session identifier"),
    db: Session = Depends(get_db)
):
    """Deletes a document from the database, disk, and FAISS vector index."""
    doc = RAGService.get_document_by_id(db=db, document_id=document_id)
    if not doc or doc.session_id != session_id:
        raise HTTPException(status_code=404, detail="Document not found.")

    success = RAGService.delete_document(db=db, document_id=document_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete document.")

    return {"status": "success", "message": f"Document '{doc.filename}' deleted successfully."}

