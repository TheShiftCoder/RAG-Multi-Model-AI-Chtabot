"""Image Analysis and Generation API Endpoints."""

import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from multimodal_ai_companion.backend.core.config import settings
from multimodal_ai_companion.backend.core.security import validate_file_upload, generate_secure_storage_path
from multimodal_ai_companion.backend.schemas.image import (
    ImageAnalyzeRequest,
    ImageAnalyzeResponse,
    ImageGenerateRequest,
    ImageGenerateResponse,
)
from multimodal_ai_companion.backend.services.multimodal_service import MultimodalService

router = APIRouter(prefix="/image", tags=["Images"])


@router.post("/upload")
async def upload_image_endpoint(file: UploadFile = File(...)):
    """Uploads an image (JPEG, PNG, WEBP) for vision analysis."""
    ext = validate_file_upload(file, settings.ALLOWED_IMAGE_EXTENSIONS)
    storage_dir = Path(settings.UPLOAD_PATH)
    target_path, unique_name = generate_secure_storage_path(storage_dir, file.filename)

    with open(target_path, "wb") as f:
        while content := await file.read(1024 * 1024):
            f.write(content)

    return {
        "status": "success",
        "image_path": str(target_path),
        "filename": unique_name,
        "format": ext,
    }


@router.post("/analyze", response_model=ImageAnalyzeResponse)
def analyze_image_endpoint(request: ImageAnalyzeRequest):
    """Analyzes an uploaded image with a visual prompt."""
    try:
        result = MultimodalService.analyze_image(
            image_path=request.image_path,
            prompt=request.prompt
        )
        return ImageAnalyzeResponse(
            description=result.get("description", ""),
            labels=result.get("labels", []),
            width=result.get("width"),
            height=result.get("height")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image inspection failed: {str(e)}")


@router.post("/generate", response_model=ImageGenerateResponse)
def generate_image_endpoint(request: ImageGenerateRequest):
    """Generates an image from a prompt."""
    try:
        filename = MultimodalService.generate_image(
            prompt=request.prompt,
            session_id=request.session_id
        )
        return ImageGenerateResponse(
            image_url=f"/api/image/file/{filename}",
            prompt=request.prompt,
            status="generated"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


@router.get("/file/{filename}")
def stream_image_file(filename: str):
    """Streams a generated image file."""
    clean_name = os.path.basename(filename)
    file_path = Path(settings.GENERATED_PATH) / clean_name
    if not file_path.exists():
        # Check upload path as well
        file_path = Path(settings.UPLOAD_PATH) / clean_name
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found.")

    ext = clean_name.split(".")[-1].lower()
    media_type = f"image/{'jpeg' if ext in ['jpg', 'jpeg'] else ext}"
    return FileResponse(path=str(file_path), media_type=media_type)

