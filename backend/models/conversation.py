"""SQLAlchemy Model for Conversations."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import relationship
from multimodal_ai_companion.backend.database.session import Base


class Conversation(Base):
    """Conversation model representing a dialogue thread."""

    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(64), index=True, nullable=False)
    title = Column(String(255), nullable=False, default="New Conversation")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Cascading relationships
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )
    feedback = relationship(
        "Feedback",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id} session={self.session_id} title='{self.title}'>"

