"""Chat and Conversation API Endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from multimodal_ai_companion.backend.database.session import get_db
from multimodal_ai_companion.backend.models.conversation import Conversation
from multimodal_ai_companion.backend.models.message import Message
from multimodal_ai_companion.backend.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationSchema,
    ConversationSummarySchema,
)
from multimodal_ai_companion.backend.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
def send_chat_message(request: ChatRequest, db: Session = Depends(get_db)):
    """Processes a user message and returns the multimodal assistant response."""
    return ChatService.handle_chat_turn(db=db, request=request)


@router.get("/conversations", response_model=List[ConversationSummarySchema])
def list_conversations(
    session_id: str = Query(..., description="Unique session ID"),
    db: Session = Depends(get_db)
):
    """Retrieves all conversation threads belonging to a session."""
    convs = (
        db.query(Conversation)
        .filter(Conversation.session_id == session_id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    result = []
    for c in convs:
        result.append(
            ConversationSummarySchema(
                id=c.id,
                session_id=c.session_id,
                title=c.title,
                created_at=c.created_at,
                updated_at=c.updated_at,
                message_count=len(c.messages),
            )
        )
    return result


@router.get("/conversations/{conversation_id}", response_model=ConversationSchema)
def get_conversation_details(
    conversation_id: str,
    session_id: str = Query(..., description="Session identifier for authorization"),
    db: Session = Depends(get_db)
):
    """Fetches a specific conversation and all of its messages."""
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.session_id == session_id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied.")
    return conv


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    session_id: str = Query(..., description="Session identifier"),
    db: Session = Depends(get_db)
):
    """Deletes a single conversation and its cascading messages."""
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.session_id == session_id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    db.delete(conv)
    db.commit()
    return {"status": "success", "message": "Conversation deleted successfully."}


@router.delete("/conversations/session/clear")
def clear_all_session_conversations(
    session_id: str = Query(..., description="Session identifier"),
    db: Session = Depends(get_db)
):
    """Clears all conversation threads for the active session."""
    convs = db.query(Conversation).filter(Conversation.session_id == session_id).all()
    count = len(convs)
    for c in convs:
        db.delete(c)
    db.commit()
    return {"status": "success", "message": f"Cleared {count} conversations."}

