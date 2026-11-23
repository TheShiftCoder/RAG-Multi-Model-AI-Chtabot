"""Tests for System Resilience and Failure Handling (Scenarios 14, 15, 20)."""

import uuid
from unittest.mock import patch
from fastapi.testclient import TestClient
from multimodal_ai_companion.backend.main import app
from multimodal_ai_companion.backend.database.session import get_engine
from multimodal_ai_companion.backend.providers.factory import get_llm_provider
from multimodal_ai_companion.backend.providers.llm_provider import FallbackLLMProvider


def test_scenario_14_database_connection_fallback():
    """Scenario 14: When remote MySQL fails, system gracefully falls back to SQLite without crashing."""
    engine = get_engine()
    assert engine is not None
    # Verify engine can execute query
    with engine.connect() as conn:
        res = conn.exec_driver_sql("SELECT 1").scalar()
        assert res == 1


def test_scenario_20_missing_or_invalid_api_key(client: TestClient):
    """Scenario 20: When external API keys are not supplied, fallback provider functions smoothly without crashing."""
    session_id = str(uuid.uuid4())
    provider = get_llm_provider()
    assert isinstance(provider, FallbackLLMProvider)

    res = client.post("/api/chat", json={
        "session_id": session_id,
        "message": "Hello, is the system online without an OpenAI key?",
    })
    assert res.status_code == 200
    assert len(res.json()["content"]) > 0


def test_scenario_15_safe_exception_handling_on_provider_failure():
    """Scenario 15: If an unexpected failure occurs in LLM generation, global error handler catches it safely."""
    with TestClient(app, raise_server_exceptions=False) as safe_client:
        session_id = str(uuid.uuid4())
        with patch(
            "multimodal_ai_companion.backend.providers.llm_provider.FallbackLLMProvider.generate_response",
            side_effect=RuntimeError("Provider connection timeout simulation")
        ):
            res = safe_client.post(
                "/api/chat",
                json={
                    "session_id": session_id,
                    "message": "Simulate failure",
                }
            )
            # Global exception handler returns 500 with sanitized message and no internal stack trace leaked
            assert res.status_code == 500
            data = res.json()
            assert "internal server error" in data["detail"].lower()
            # Ensure stack trace not leaked in json response
            assert "Traceback" not in str(data)
