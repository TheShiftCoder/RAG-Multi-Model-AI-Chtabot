"""Pydantic Schemas for Audio STT and TTS."""

from typing import Optional
from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    """Schema returned after transcribing audio."""
    text: str
    duration_seconds: Optional[float] = None
    language: Optional[str] = "en"
    audio_path: str


class AudioGenerateRequest(BaseModel):
    """Schema for requesting Text-To-Speech generation."""
    text: str = Field(..., min_length=1, description="Text to synthesize")
    session_id: Optional[str] = None


class AudioGenerateResponse(BaseModel):
    """Schema returned after TTS synthesis."""
    audio_url: str
    format: str = "mp3"
    message: str = "Audio generated successfully"

