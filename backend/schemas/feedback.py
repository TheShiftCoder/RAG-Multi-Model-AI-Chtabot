"""Pydantic Schemas for Feedback."""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class FeedbackCreate(BaseModel):
    """Schema for submitting user feedback."""
    conversation_id: str
    message_id: Optional[str] = None
    rating: int = Field(..., ge=-1, le=5, description="Rating (-1 to 5)")
    feedback_text: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response returned upon saving feedback."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    rating: int
    feedback_text: Optional[str] = None
    created_at: datetime
