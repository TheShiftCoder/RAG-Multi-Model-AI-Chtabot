"""Provider Factory Module.

Instantiates configured LLM, Embedding, Vision, STT, TTS, and Image Gen
providers based on application settings and available API keys.
"""

from multimodal_ai_companion.backend.core.config import settings
from multimodal_ai_companion.backend.providers.base import (
    BaseLLMProvider,
    BaseEmbeddingProvider,
    BaseVisionProvider,
    BaseAudioSTTProvider,
    BaseAudioTTSProvider,
    BaseImageGenProvider,
)
from multimodal_ai_companion.backend.providers.llm_provider import (
    OpenAILLMProvider,
    FallbackLLMProvider,
)
from multimodal_ai_companion.backend.providers.embedding_provider import (
    OpenAIEmbeddingProvider,
    LocalDeterministicEmbeddingProvider,
)
from multimodal_ai_companion.backend.providers.vision_provider import (
    OpenAIVisionProvider,
    FallbackVisionProvider,
)
from multimodal_ai_companion.backend.providers.audio_provider import (
    OpenAIAudioTTSProvider,
    GTTSAudioTTSProvider,
    FallbackAudioTTSProvider,
    OpenAIAudioSTTProvider,
    FallbackAudioSTTProvider,
)
from multimodal_ai_companion.backend.providers.image_gen_provider import (
    OpenAIImageGenProvider,
    LocalPlaceholderImageGenProvider,
)


def get_llm_provider() -> BaseLLMProvider:
    if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        return OpenAILLMProvider(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL)
    return FallbackLLMProvider()


def get_embedding_provider() -> BaseEmbeddingProvider:
    if settings.EMBEDDING_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        return OpenAIEmbeddingProvider(api_key=settings.OPENAI_API_KEY)
    return LocalDeterministicEmbeddingProvider()


def get_vision_provider() -> BaseVisionProvider:
    if settings.VISION_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        return OpenAIVisionProvider(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL)
    return FallbackVisionProvider()


def get_stt_provider() -> BaseAudioSTTProvider:
    if settings.STT_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        return OpenAIAudioSTTProvider(api_key=settings.OPENAI_API_KEY)
    return FallbackAudioSTTProvider()


def get_tts_provider() -> BaseAudioTTSProvider:
    if settings.TTS_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        return OpenAIAudioTTSProvider(api_key=settings.OPENAI_API_KEY)
    if settings.TTS_PROVIDER == "gtts":
        return GTTSAudioTTSProvider()
    return FallbackAudioTTSProvider()


def get_image_gen_provider() -> BaseImageGenProvider:
    if settings.IMAGE_GEN_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        return OpenAIImageGenProvider(api_key=settings.OPENAI_API_KEY)
    return LocalPlaceholderImageGenProvider()

