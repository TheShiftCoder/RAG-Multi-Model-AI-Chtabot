"""SQLAlchemy Model for User Feedback."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from multimodal_ai_companion.backend.database.session import Base


class Feedback(Base):
    """Feedback model storing user ratings and qualitative feedback."""

    __tablename__ = "feedback"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    message_id = Column(String(36), nullable=True)
    rating = Column(Integer, nullable=False)  # 1 to 5, or 1 (up) / -1 (down)
    feedback_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    conversation = relationship("Conversation", back_populates="feedback")

    def __repr__(self) -> str:
        return f"<Feedback id={self.id} rating={self.rating}>"

