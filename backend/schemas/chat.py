"""Pydantic Schemas for Chat and Conversations."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class CitationSchema(BaseModel):
    """Schema for retrieved document citations."""
    document_id: str
    filename: str
    chunk_index: int
    text_snippet: str
    score: float


class MessageSchema(BaseModel):
    """Schema for serialized messages."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    role: str
    content: str
    modality: str = "text"
    raw_metadata: Optional[str] = None
    created_at: datetime


class ConversationSchema(BaseModel):
    """Schema for serialized conversations."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[MessageSchema] = []


class ConversationSummarySchema(BaseModel):
    """Summary representation of a conversation for list views."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ChatRequest(BaseModel):
    """Request payload for sending a chat message."""
    session_id: str = Field(..., description="Unique session identifier for isolation")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID if continuing")
    message: str = Field(..., min_length=1, description="User prompt text")
    modality: str = Field("text", description="Primary modality: text, image, audio, document")
    media_url: Optional[str] = Field(None, description="Optional path/URL of uploaded media")
    request_audio: bool = Field(False, description="Whether to generate an audio version of the response")
    request_image: bool = Field(False, description="Whether the user explicitly requests an image response")


class ChatResponse(BaseModel):
    """Response payload returned by the chat API."""
    conversation_id: str
    message_id: str
    role: str = "assistant"
    content: str
    modality: str = "text"
    audio_url: Optional[str] = None
    image_url: Optional[str] = None
    citations: List[CitationSchema] = []
    is_crisis: bool = False
    is_rag_used: bool = False
    created_at: datetime
