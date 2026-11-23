"""Multimodal Service Module.

Handles vision understanding, speech transcription, text-to-speech generation,
and image synthesis tasks.
"""

import os
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from multimodal_ai_companion.backend.core.config import settings
from multimodal_ai_companion.backend.providers.factory import (
    get_vision_provider,
    get_stt_provider,
    get_tts_provider,
    get_image_gen_provider,
)


class MultimodalService:
    """Orchestrates image inspection, audio transcription, TTS, and image generation."""

    @classmethod
    def analyze_image(cls, image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Analyzes an image file using the configured vision provider."""
        vision_provider = get_vision_provider()
        user_prompt = prompt or "Describe this image in detail and extract key information."
        return vision_provider.analyze_image(image_path, user_prompt)

    @classmethod
    def transcribe_audio(cls, audio_path: str) -> Dict[str, Any]:
        """Converts speech audio into text using the configured STT provider."""
        stt_provider = get_stt_provider()
        return stt_provider.transcribe(audio_path)

    @classmethod
    def generate_speech(cls, text: str, session_id: Optional[str] = None) -> str:
        """Synthesizes text to speech and returns the relative file path."""
        tts_provider = get_tts_provider()
        gen_dir = Path(settings.GENERATED_PATH)
        filename = f"tts_{uuid.uuid4().hex[:10]}.mp3"
        output_path = str(gen_dir / filename)

        final_path = tts_provider.generate_speech(text, output_path)
        # Return filename or relative path accessible via static endpoint
        return os.path.basename(final_path)

    @classmethod
    def generate_image(cls, prompt: str, session_id: Optional[str] = None) -> str:
        """Synthesizes an image and returns the relative file path."""
        image_gen_provider = get_image_gen_provider()
        gen_dir = Path(settings.GENERATED_PATH)
        filename = f"img_{uuid.uuid4().hex[:10]}.png"
        output_path = str(gen_dir / filename)

        final_path = image_gen_provider.generate_image(prompt, output_path)
        return os.path.basename(final_path)

