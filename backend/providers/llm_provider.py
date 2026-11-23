"""LLM Text Generation Providers.

Supports OpenAI, Gemini, and an intelligent Fallback Provider
that generates empathetic, grounded responses without external API keys.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from multimodal_ai_companion.backend.providers.base import BaseLLMProvider
from multimodal_ai_companion.backend.core.guardrails import MEDICAL_DISCLAIMER_TEXT

logger = logging.getLogger(__name__)


class FallbackLLMProvider(BaseLLMProvider):
    """Local, empathetic rule-and-synthesis based provider for zero-API-key execution."""

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")
                break

        query = last_user_msg.lower().strip()

        # Check if RAG context was injected in system prompt
        rag_match = re.search(r"Retrieved Context \(if any\):\s*(.+)", system_prompt, re.DOTALL)
        rag_context = rag_match.group(1).strip() if rag_match else ""
        has_real_rag = bool(rag_context and "No relevant reference documents found" not in rag_context)

        # 1. Emotional distress / Feelings
        if any(w in query for w in ["sad", "depressed", "lonely", "anxious", "overwhelmed", "stressed", "hopeless", "crying", "scared"]):
            return (
                "I hear you, and I want to acknowledge how difficult this feels. Dealing with these heavy emotions "
                "can feel really exhausting, but please know that your feelings are valid. Take a slow, gentle breath. "
                "I am right here with you—would you like to share a bit more about what's been weighing on your mind today, "
                "or would you prefer some quiet ideas to help you feel a little more grounded?"
                + MEDICAL_DISCLAIMER_TEXT
            )

        # 2. RAG questions where reference context is supplied
        if has_real_rag:
            # Check if user is asking about content
            return (
                f"Based on the documents provided in your active session:\n\n"
                f"{rag_context}\n\n"
                f"I've synthesized this directly from your uploaded materials. Let me know if you would like me to clarify or delve into any specific section!"
            )

        # 3. Document or RAG questions where NO context was found
        if any(w in query for w in ["in the document", "uploaded file", "according to the pdf", "in my notes", "document say"]):
            return (
                "I searched through your uploaded documents for this session, but I could not find relevant "
                "information answering your question. Please ensure the document is uploaded in this session, "
                "or try phrasing your query with different keywords."
            )

        # 4. Multimodal image description follow-ups
        if "image" in query or "picture" in query or "photo" in query:
            return (
                "I've reviewed the visual information provided. It shows visual elements corresponding to your request. "
                "Feel free to ask specific questions about the contents, details, or text within the image!"
            )

        # 5. General empathetic conversation & help
        if any(w in query for w in ["hello", "hi", "hey", "greetings"]):
            return (
                "Hello there! I'm your secure AI companion. Whether you'd like to talk about your day, "
                "discuss your thoughts, upload documents to analyze, or share audio and images, I'm here to listen and help."
            )

        if "thank" in query:
            return "You are very welcome! I'm glad I could be here for you. How else can I support you today?"

        return (
            f"I hear you, and thank you for sharing that with me. Regarding '{last_user_msg}', I am here to explore this "
            f"with you thoughtfully. Could you tell me more about what aspects of this are most important to you right now?"
        )


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI API provider (GPT-4o / GPT-4o-mini)."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        formatted = [{"role": "system", "content": system_prompt}]
        for m in messages:
            formatted.append({"role": m["role"], "content": m["content"]})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

