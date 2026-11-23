"""Vision and Image Understanding Providers."""

import base64
import os
import logging
from typing import Dict, Any
from PIL import Image
from multimodal_ai_companion.backend.providers.base import BaseVisionProvider

logger = logging.getLogger(__name__)


class FallbackVisionProvider(BaseVisionProvider):
    """Local image inspector using Pillow when external vision APIs are unconfigured."""

    def analyze_image(self, image_path: str, prompt: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            return {
                "description": "Image file not found on server.",
                "labels": ["missing_file"],
                "width": 0,
                "height": 0
            }

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                format_name = img.format or "Unknown"
                mode = img.mode

                # Basic heuristic labels based on aspect ratio
                aspect = width / max(1, height)
                orientation = "landscape" if aspect > 1.2 else ("portrait" if aspect < 0.8 else "square")

                description = (
                    f"Visual Inspection of Uploaded Image:\n"
                    f"- Format: {format_name}\n"
                    f"- Dimensions: {width} x {height} pixels ({orientation} orientation)\n"
                    f"- Color Mode: {mode}\n\n"
                    f"In response to your query ('{prompt}'): "
                    f"The image was validated and processed successfully. In full cloud mode with an active Vision API key, "
                    f"dense object labels and OCR text detection are rendered here."
                )

                return {
                    "description": description,
                    "labels": [format_name.lower(), orientation, f"{width}x{height}"],
                    "width": width,
                    "height": height
                }
        except Exception as e:
            logger.error(f"Failed to inspect image {image_path}: {e}")
            return {
                "description": f"Error analyzing image: {str(e)}",
                "labels": ["error"],
                "width": 0,
                "height": 0
            }


class OpenAIVisionProvider(BaseVisionProvider):
    """OpenAI Vision Provider using GPT-4o."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def analyze_image(self, image_path: str, prompt: str) -> Dict[str, Any]:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        ext = image_path.split(".")[-1].lower()
        mime = f"image/{'jpeg' if ext in ['jpg', 'jpeg'] else ext}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"}
                        }
                    ]
                }
            ],
            max_tokens=600
        )
        description = response.choices[0].message.content or "No visual description generated."
        return {
            "description": description,
            "labels": ["vision-api"],
            "width": None,
            "height": None
        }

