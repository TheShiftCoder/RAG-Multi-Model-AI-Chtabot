"""SQLAlchemy Model for Messages."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from multimodal_ai_companion.backend.database.session import Base


class Message(Base):
    """Message model representing a user or assistant response in a conversation."""

    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    modality = Column(String(30), nullable=False, default="text")  # 'text', 'image', 'audio', 'document'
    raw_metadata = Column(Text, nullable=True)  # JSON-encoded citations, media URLs, audio paths
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role} modality={self.modality}>"

