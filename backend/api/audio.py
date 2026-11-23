"""Audio Processing API Endpoints."""

import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse

from multimodal_ai_companion.backend.core.config import settings
from multimodal_ai_companion.backend.core.security import validate_file_upload, generate_secure_storage_path
from multimodal_ai_companion.backend.schemas.audio import (
    TranscriptionResponse,
    AudioGenerateRequest,
    AudioGenerateResponse,
)
from multimodal_ai_companion.backend.services.multimodal_service import MultimodalService

router = APIRouter(prefix="/audio", tags=["Audio"])


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_endpoint(
    file: UploadFile = File(...)
):
    """Transcribes an uploaded audio file into text."""
    ext = validate_file_upload(file, settings.ALLOWED_AUDIO_EXTENSIONS)
    storage_dir = Path(settings.UPLOAD_PATH)
    target_path, _ = generate_secure_storage_path(storage_dir, file.filename)

    with open(target_path, "wb") as f:
        while content := await file.read(1024 * 1024):
            f.write(content)

    try:
        result = MultimodalService.transcribe_audio(str(target_path))
        return TranscriptionResponse(
            text=result.get("text", ""),
            duration_seconds=result.get("duration_seconds"),
            language=result.get("language", "en"),
            audio_path=str(target_path)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {str(e)}")


@router.post("/generate", response_model=AudioGenerateResponse)
def generate_speech_endpoint(request: AudioGenerateRequest):
    """Synthesizes text into speech and returns a streamable audio URL."""
    try:
        filename = MultimodalService.generate_speech(text=request.text, session_id=request.session_id)
        return AudioGenerateResponse(
            audio_url=f"/api/audio/file/{filename}",
            format=filename.split(".")[-1],
            message="Audio speech generated successfully."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")


@router.get("/file/{filename}")
def stream_audio_file(filename: str):
    """Streams a generated audio file."""
    # Sanitize filename
    clean_name = os.path.basename(filename)
    file_path = Path(settings.GENERATED_PATH) / clean_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    media_type = "audio/mpeg" if clean_name.endswith(".mp3") else "audio/wav"
    return FileResponse(path=str(file_path), media_type=media_type)

