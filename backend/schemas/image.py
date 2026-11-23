"""Pydantic Schemas for Image Analysis and Generation."""

from typing import Optional
from pydantic import BaseModel, Field


class ImageAnalyzeRequest(BaseModel):
    """Schema for requesting image inspection or Q&A."""
    image_path: str
    prompt: Optional[str] = "Describe this image in detail and extract key information."


class ImageAnalyzeResponse(BaseModel):
    """Response returned after vision model analysis."""
    description: str
    labels: list[str] = []
    width: Optional[int] = None
    height: Optional[int] = None


class ImageGenerateRequest(BaseModel):
    """Schema for requesting image generation."""
    prompt: str = Field(..., min_length=2, description="Visual description prompt")
    session_id: Optional[str] = None


class ImageGenerateResponse(BaseModel):
    """Response returned with generated image details."""
    image_url: str
    prompt: str
    status: str = "generated"

