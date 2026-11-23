"""Test configuration and shared fixtures."""

import os
import pytest
from fastapi.testclient import TestClient
from multimodal_ai_companion.backend.main import app
from multimodal_ai_companion.backend.database.session import SessionLocal, Base, engine


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Ensure database schema is created."""
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def client():
    """Provides FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_session():
    """Provides isolated SQLAlchemy session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

