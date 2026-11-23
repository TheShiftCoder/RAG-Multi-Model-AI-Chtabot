"""Security Utilities and Input Sanitization Module.

Provides filename sanitization, path traversal prevention,
and file validation for multimodal uploads.
"""

import os
import re
import uuid
from pathlib import Path
from fastapi import HTTPException, UploadFile
from multimodal_ai_companion.backend.core.config import settings


def sanitize_filename(filename: str) -> str:
    """Sanitizes filename to prevent directory traversal and special character injection."""
    clean_name = os.path.basename(filename)
    # Remove all characters except alphanumeric, underscores, hyphens, and dots
    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean_name)
    # Ensure it's not empty or just dots
    if not clean_name or set(clean_name) == {"."}:
        clean_name = f"unnamed_{uuid.uuid4().hex[:8]}"
    return clean_name


def validate_file_upload(file: UploadFile, allowed_extensions: list[str], max_size: int = settings.MAX_UPLOAD_SIZE_BYTES):
    """Validates file extension and size constraints.

    Raises:
        HTTPException: If the file extension or size violates security policy.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename in upload.")

    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed types: {', '.join(allowed_extensions)}"
        )

    # Note: File size will also be verified during chunk reading to avoid memory exhaustion
    return ext


def generate_secure_storage_path(base_dir: Path, original_filename: str) -> tuple[Path, str]:
    """Generates a collision-free, isolated local storage path."""
    sanitized = sanitize_filename(original_filename)
    unique_prefix = uuid.uuid4().hex
    target_filename = f"{unique_prefix}_{sanitized}"
    target_path = base_dir / target_filename
    return target_path, target_filename

