"""Abstract Base Classes for Multimodal Model Providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLMProvider(ABC):
    """Abstract interface for text generation models."""

    @abstractmethod
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generates a text completion based on conversation history and system instructions."""
        pass


class BaseEmbeddingProvider(ABC):
    """Abstract interface for generating vector embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimensionality of the generated vectors."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a batch of text chunks."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generates an embedding vector for a single search query."""
        pass


class BaseVisionProvider(ABC):
    """Abstract interface for image understanding and visual Q&A."""

    @abstractmethod
    def analyze_image(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Inspects an image file and answers queries about its contents."""
        pass


class BaseAudioSTTProvider(ABC):
    """Abstract interface for Speech-to-Text transcription."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribes an audio recording into text."""
        pass


class BaseAudioTTSProvider(ABC):
    """Abstract interface for Text-to-Speech generation."""

    @abstractmethod
    def generate_speech(self, text: str, output_path: str) -> str:
        """Synthesizes speech audio from text and saves to output_path."""
        pass


class BaseImageGenProvider(ABC):
    """Abstract interface for Text-to-Image generation."""

    @abstractmethod
    def generate_image(self, prompt: str, output_path: str) -> str:
        """Generates an image from a prompt and saves to output_path."""
        pass

