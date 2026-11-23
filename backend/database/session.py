"""Database Session & Connection Management Module.

Configures SQLAlchemy engine, connection pooling, and session maker.
Supports MySQL as the primary database with automatic graceful fallback
to local SQLite for offline development and isolated automated testing.
"""

import logging
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from multimodal_ai_companion.backend.core.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


def get_engine():
    """Initializes SQLAlchemy engine with MySQL or SQLite fallback."""
    db_url = settings.DATABASE_URL
    try:
        if "sqlite" in db_url:
            engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False}
            )
        else:
            engine = create_engine(
                db_url,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={"connect_timeout": 3}
            )
        # Verify connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Database connected successfully: {engine.url.render_as_string(hide_password=True)}")
        return engine
    except Exception as e:
        logger.warning(
            f"Failed to connect to primary database at {db_url} ({e}). "
            f"Falling back to local SQLite at {settings.SQLITE_FALLBACK_URL}."
        )
        engine = create_engine(
            settings.SQLITE_FALLBACK_URL,
            connect_args={"check_same_thread": False}
        )
        return engine


engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for obtaining a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

