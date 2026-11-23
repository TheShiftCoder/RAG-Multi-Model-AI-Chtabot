"""Safety and Guardrails Service."""

from typing import Tuple, Optional, List, Dict
from multimodal_ai_companion.backend.core.guardrails import (
    check_for_crisis,
    check_for_prompt_injection,
    sanitize_retrieved_context,
)


class SafetyService:
    """Evaluates user requests and external inputs against AI safety guardrails."""

    @staticmethod
    def evaluate_input_safety(text: str) -> Tuple[bool, Optional[str]]:
        """Checks for immediate crisis, self-harm, or severe safety violations."""
        return check_for_crisis(text)

    @staticmethod
    def inspect_prompt_injection(text: str) -> Tuple[bool, str]:
        """Detects prompt injection attempts and neutralizes override patterns."""
        return check_for_prompt_injection(text)

    @staticmethod
    def format_safe_rag_context(chunks: List[Dict]) -> str:
        """Wraps retrieved chunks in protective delimiters."""
        return sanitize_retrieved_context(chunks)

