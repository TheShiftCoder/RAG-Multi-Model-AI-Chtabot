"""Emotional Support and Empathetic Conversation Service."""

import re
from typing import Dict, Any


class EmotionalSupportService:
    """Detects emotional cues and generates empathetic follow-up suggestions."""

    EMOTIONAL_CUES = {
        "anxiety": [r"\banxious\b", r"\bworry\b", r"\bpanic\b", r"\bnervous\b", r"\bstressed\b"],
        "sadness": [r"\bsad\b", r"\bgrief\b", r"\bcrying\b", r"\bheartbroken\b", r"\bdepressed\b", r"\blonely\b"],
        "anger": [r"\bangry\b", r"\bfurious\b", r"\bfrustrated\b", r"\birritated\b", r"\bmad\b"],
        "exhaustion": [r"\btired\b", r"\bburned\s*out\b", r"\bexhausted\b", r"\boverwhelmed\b", r"\bdrained\b"],
        "joy": [r"\bhappy\b", r"\bexcited\b", r"\bdelighted\b", r"\bproud\b", r"\bgrateful\b"]
    }

    @classmethod
    def detect_emotional_state(cls, text: str) -> Dict[str, Any]:
        """Identifies emotional indicators in the text."""
        lower = text.lower()
        detected = []
        for emotion, patterns in cls.EMOTIONAL_CUES.items():
            for p in patterns:
                if re.search(p, lower):
                    detected.append(emotion)
                    break

        return {
            "has_emotional_cue": len(detected) > 0,
            "emotions": detected
        }

