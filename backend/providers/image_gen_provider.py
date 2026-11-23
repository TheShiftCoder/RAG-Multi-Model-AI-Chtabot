"""Image Generation Providers."""

import os
import logging
from PIL import Image, ImageDraw, ImageFont
from multimodal_ai_companion.backend.providers.base import BaseImageGenProvider

logger = logging.getLogger(__name__)


class LocalPlaceholderImageGenProvider(BaseImageGenProvider):
    """Generates clean graphical visual cards using Pillow locally."""

    def generate_image(self, prompt: str, output_path: str) -> str:
        width, height = 768, 512
        # Create an attractive gradient image
        image = Image.new("RGB", (width, height), color=(30, 41, 59))
        draw = ImageDraw.Draw(image)

        # Draw decorative gradient / concentric shapes
        for r in range(180, 0, -20):
            color = (30 + int(r * 0.4), 60 + int(r * 0.5), 100 + int(r * 0.7))
            draw.ellipse(
                [(width // 2 - r * 2, height // 2 - r), (width // 2 + r * 2, height // 2 + r)],
                outline=color,
                width=3
            )

        # Draw banner rectangle for prompt
        box_y1 = height - 120
        draw.rectangle([(40, box_y1), (width - 40, height - 30)], fill=(15, 23, 42), outline=(99, 102, 241), width=2)

        # Wrap and draw text
        prompt_preview = f"Generated Artwork: \"{prompt[:60]}...\"" if len(prompt) > 60 else f"Generated Artwork: \"{prompt}\""
        draw.text((60, box_y1 + 15), prompt_preview, fill=(248, 250, 252))
        draw.text((60, box_y1 + 45), "Secure Multimodal AI Companion • Image Synthesis Layer", fill=(148, 163, 184))

        # Ensure parent directory exists and save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path, "PNG")
        return output_path


class OpenAIImageGenProvider(BaseImageGenProvider):
    """OpenAI DALL-E 3 Image Generation Provider."""

    def __init__(self, api_key: str, model: str = "dall-e-3"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_image(self, prompt: str, output_path: str) -> str:
        import requests
        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        image_url = response.data[0].url
        res = requests.get(image_url, timeout=30)
        res.raise_for_status()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(res.content)
        return output_path

