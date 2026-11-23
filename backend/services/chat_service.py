"""Chat Service Orchestration Module.

Combines conversation history, safety guardrails, RAG context retrieval,
multimodal input analysis, and LLM text/audio/image generation.
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException

from multimodal_ai_companion.backend.core.config import settings
from multimodal_ai_companion.backend.core.guardrails import (
    SYSTEM_PROMPT_TEMPLATE,
    MEDICAL_DISCLAIMER_TEXT,
)
from multimodal_ai_companion.backend.models.conversation import Conversation
from multimodal_ai_companion.backend.models.message import Message
from multimodal_ai_companion.backend.schemas.chat import (
    ChatRequest,
    ChatResponse,
    CitationSchema,
)
from multimodal_ai_companion.backend.services.safety_service import SafetyService
from multimodal_ai_companion.backend.services.rag_service import RAGService
from multimodal_ai_companion.backend.services.multimodal_service import MultimodalService
from multimodal_ai_companion.backend.providers.factory import get_llm_provider

logger = logging.getLogger(__name__)


class ChatService:
    """Coordinates conversational turns with guardrails, RAG, and multimodal features."""

    @classmethod
    def handle_chat_turn(
        cls,
        db: Session,
        request: ChatRequest
    ) -> ChatResponse:
        """Processes a complete conversational turn."""
        session_id = request.session_id.strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required.")

        # 1. Retrieve or initialize Conversation
        conversation = None
        if request.conversation_id:
            conversation = (
                db.query(Conversation)
                .filter(
                    Conversation.id == request.conversation_id,
                    Conversation.session_id == session_id,
                )
                .first()
            )
            if not conversation:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found or does not belong to this session."
                )
        else:
            # Generate title from user message preview
            title_preview = request.message.strip().split("\n")[0][:45]
            if len(request.message.strip()) > 45:
                title_preview += "..."
            conversation = Conversation(
                id=str(uuid.uuid4()),
                session_id=session_id,
                title=title_preview or "New Conversation",
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)

        # 2. Safety & Crisis Inspection (Priority 1)
        is_crisis, crisis_msg = SafetyService.evaluate_input_safety(request.message)
        if is_crisis:
            # Record user message
            user_msg = Message(
                id=str(uuid.uuid4()),
                conversation_id=conversation.id,
                role="user",
                content=request.message,
                modality=request.modality,
            )
            db.add(user_msg)

            # Record crisis response
            assistant_msg = Message(
                id=str(uuid.uuid4()),
                conversation_id=conversation.id,
                role="assistant",
                content=crisis_msg,
                modality="text",
                raw_metadata=json.dumps({"is_crisis": True}),
            )
            db.add(assistant_msg)
            conversation.updated_at = datetime.now(timezone.utc)
            db.commit()

            return ChatResponse(
                conversation_id=conversation.id,
                message_id=assistant_msg.id,
                role="assistant",
                content=crisis_msg,
                modality="text",
                citations=[],
                is_crisis=True,
                is_rag_used=False,
                created_at=assistant_msg.created_at,
            )

        # 3. Prompt injection inspection
        is_injection, sanitized_user_msg = SafetyService.inspect_prompt_injection(request.message)

        # 4. Multimodal input augmentation (Image or Audio)
        augmented_prompt = sanitized_user_msg
        if request.modality == "image" and request.media_url:
            image_analysis = MultimodalService.analyze_image(request.media_url, request.message)
            augmented_prompt = (
                f"{sanitized_user_msg}\n\n"
                f"[Attached Image Inspection Data]:\n"
                f"{image_analysis.get('description', '')}"
            )

        # 5. RAG Semantic Retrieval
        citations: List[CitationSchema] = []
        is_rag_used = False
        rag_context_str = "No relevant reference documents found."

        # Search session documents
        retrieved_chunks = RAGService.retrieve_session_context(
            session_id=session_id,
            query=request.message,
            top_k=settings.MAX_RETRIEVAL_CHUNKS,
            similarity_threshold=settings.SIMILARITY_THRESHOLD,
        )

        if retrieved_chunks:
            is_rag_used = True
            rag_context_str = SafetyService.format_safe_rag_context(retrieved_chunks)
            for c in retrieved_chunks:
                citations.append(
                    CitationSchema(
                        document_id=c.get("document_id", ""),
                        filename=c.get("filename", "Unknown"),
                        chunk_index=c.get("chunk_index", 0),
                        text_snippet=c.get("text", "")[:200] + "...",
                        score=round(c.get("score", 0.0), 3),
                    )
                )

        # 6. Build Prompt & Call LLM
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(rag_context=rag_context_str)

        # Load recent conversational history (up to last 10 messages for context)
        history_records = (
            db.query(Message)
            .filter(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.asc())
            .all()[-10:]
        )

        history_msgs = [{"role": m.role, "content": m.content} for m in history_records]
        history_msgs.append({"role": "user", "content": augmented_prompt})

        llm_provider = get_llm_provider()
        ai_reply = llm_provider.generate_response(
            messages=history_msgs,
            system_prompt=system_prompt,
        )

        # 7. Optional Multimodal Outputs (Audio TTS & Image Gen)
        audio_url: Optional[str] = None
        if request.request_audio:
            try:
                audio_filename = MultimodalService.generate_speech(ai_reply, session_id=session_id)
                audio_url = f"/api/audio/file/{audio_filename}"
            except Exception as e:
                logger.error(f"TTS synthesis error: {e}")

        image_url: Optional[str] = None
        if request.request_image or "generate an image" in request.message.lower():
            try:
                img_filename = MultimodalService.generate_image(request.message, session_id=session_id)
                image_url = f"/api/image/file/{img_filename}"
            except Exception as e:
                logger.error(f"Image generation error: {e}")

        # 8. Persist Messages in DB
        user_msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation.id,
            role="user",
            content=request.message,
            modality=request.modality,
        )
        db.add(user_msg)

        meta_dict = {
            "is_crisis": False,
            "is_rag_used": is_rag_used,
            "citations": [c.model_dump() for c in citations],
            "audio_url": audio_url,
            "image_url": image_url,
        }
        assistant_msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation.id,
            role="assistant",
            content=ai_reply,
            modality="multimodal" if (audio_url or image_url) else "text",
            raw_metadata=json.dumps(meta_dict),
        )
        db.add(assistant_msg)
        conversation.updated_at = datetime.now(timezone.utc)
        db.commit()

        return ChatResponse(
            conversation_id=conversation.id,
            message_id=assistant_msg.id,
            role="assistant",
            content=ai_reply,
            modality="multimodal" if (audio_url or image_url) else "text",
            audio_url=audio_url,
            image_url=image_url,
            citations=citations,
            is_crisis=False,
            is_rag_used=is_rag_used,
            created_at=assistant_msg.created_at,
        )

