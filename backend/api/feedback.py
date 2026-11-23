"""User Feedback API Endpoints."""

import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from multimodal_ai_companion.backend.database.session import get_db
from multimodal_ai_companion.backend.models.feedback import Feedback
from multimodal_ai_companion.backend.models.conversation import Conversation
from multimodal_ai_companion.backend.schemas.feedback import FeedbackCreate, FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackCreate, db: Session = Depends(get_db)):
    """Records user rating and comments on conversation turns."""
    conv = db.query(Conversation).filter(Conversation.id == request.conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    fb = Feedback(
        id=str(uuid.uuid4()),
        conversation_id=request.conversation_id,
        message_id=request.message_id,
        rating=request.rating,
        feedback_text=request.feedback_text
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)

    return fb

