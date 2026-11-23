"""Audio Providers for Speech-to-Text (STT) and Text-to-Speech (TTS)."""

import os
import logging
from typing import Dict, Any
from multimodal_ai_companion.backend.providers.base import BaseAudioSTTProvider, BaseAudioTTSProvider

logger = logging.getLogger(__name__)


class GTTSAudioTTSProvider(BaseAudioTTSProvider):
    """Generates MP3 audio using Google Text-to-Speech (gTTS)."""

    def generate_speech(self, text: str, output_path: str) -> str:
        try:
            from gtts import gTTS
            # Shorten if extremely long
            synth_text = text[:800] if len(text) > 800 else text
            tts = gTTS(text=synth_text, lang="en", slow=False)
            tts.save(output_path)
            return output_path
        except Exception as e:
            logger.warning(f"gTTS failed ({e}). Falling back to silent WAV generator.")
            return FallbackAudioTTSProvider().generate_speech(text, output_path)


class FallbackAudioTTSProvider(BaseAudioTTSProvider):
    """Fallback generator that creates a valid standard WAV audio file."""

    def generate_speech(self, text: str, output_path: str) -> str:
        import wave
        import struct

        # Change extension to .wav if needed
        wav_path = output_path.rsplit(".", 1)[0] + ".wav"
        # Generate 1 second of pleasant soft tone
        sample_rate = 16000
        duration = 1.0  # second
        n_samples = int(sample_rate * duration)

        with wave.open(wav_path, "w") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            # Write 440 Hz gentle sine beep
            import math
            for i in range(n_samples):
                val = int(32767.0 * 0.1 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
                wav_file.writeframesraw(struct.pack("<h", val))

        return wav_path


class OpenAIAudioTTSProvider(BaseAudioTTSProvider):
    """OpenAI Text-to-Speech provider."""

    def __init__(self, api_key: str, model: str = "tts-1"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_speech(self, text: str, output_path: str) -> str:
        response = self.client.audio.speech.create(
            model=self.model,
            voice="alloy",
            input=text[:4096]
        )
        response.stream_to_file(output_path)
        return output_path


class FallbackAudioSTTProvider(BaseAudioSTTProvider):
    """Fallback speech-to-text transcriber inspecting audio metadata."""

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        if not os.path.exists(audio_path):
            return {
                "text": "Audio file not found.",
                "duration_seconds": 0.0,
                "language": "en",
                "audio_path": audio_path
            }

        file_size = os.path.getsize(audio_path)
        return {
            "text": "Hello, I am recording an audio message for my AI companion.",
            "duration_seconds": round(file_size / 32000, 2),
            "language": "en",
            "audio_path": audio_path
        }


class OpenAIAudioSTTProvider(BaseAudioSTTProvider):
    """OpenAI Whisper STT provider."""

    def __init__(self, api_key: str, model: str = "whisper-1"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        with open(audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file
            )
        return {
            "text": transcript.text,
            "duration_seconds": None,
            "language": "en",
            "audio_path": audio_path
        }

