"""Tests for Conversations and Context Management (Scenarios 1, 2, 16, 17)."""

import uuid
from fastapi.testclient import TestClient


def test_scenario_01_normal_text_conversation(client: TestClient):
    """Scenario 1: Normal text conversation creates conversation and returns response."""
    session_id = str(uuid.uuid4())
    payload = {
        "session_id": session_id,
        "message": "Hello, how are you today?",
        "modality": "text",
        "request_audio": False,
        "request_image": False,
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert "content" in data
    assert len(data["content"]) > 0
    assert data["role"] == "assistant"
    assert data["is_crisis"] is False


def test_scenario_02_follow_up_question_using_context(client: TestClient):
    """Scenario 2: Follow-up question preserves conversational context and updates conversation."""
    session_id = str(uuid.uuid4())
    # Turn 1
    res1 = client.post("/api/chat", json={
        "session_id": session_id,
        "message": "My favorite color is emerald green.",
    })
    assert res1.status_code == 200
    conv_id = res1.json()["conversation_id"]

    # Turn 2: Follow-up in same conversation
    res2 = client.post("/api/chat", json={
        "session_id": session_id,
        "conversation_id": conv_id,
        "message": "Do you remember what my favorite color is?",
    })
    assert res2.status_code == 200
    assert res2.json()["conversation_id"] == conv_id

    # Verify conversation history has both user and assistant messages
    res_details = client.get(f"/api/chat/conversations/{conv_id}?session_id={session_id}")
    assert res_details.status_code == 200
    messages = res_details.json()["messages"]
    assert len(messages) >= 4  # 2 user messages + 2 assistant messages


def test_scenario_16_conversation_deletion(client: TestClient):
    """Scenario 16: Deleting a conversation removes it and its messages completely."""
    session_id = str(uuid.uuid4())
    res = client.post("/api/chat", json={
        "session_id": session_id,
        "message": "Testing conversation deletion.",
    })
    conv_id = res.json()["conversation_id"]

    # Delete conversation
    del_res = client.delete(f"/api/chat/conversations/{conv_id}?session_id={session_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

    # Verify fetching returns 404
    get_res = client.get(f"/api/chat/conversations/{conv_id}?session_id={session_id}")
    assert get_res.status_code == 404


def test_scenario_17_cross_session_data_isolation(client: TestClient):
    """Scenario 17: Session A cannot access or delete conversations belonging to Session B."""
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())

    # Create conversation in Session A
    res_a = client.post("/api/chat", json={
        "session_id": session_a,
        "message": "Secret data for Session A",
    })
    conv_id_a = res_a.json()["conversation_id"]

    # Session B attempts to access Session A conversation
    leak_check = client.get(f"/api/chat/conversations/{conv_id_a}?session_id={session_b}")
    assert leak_check.status_code == 404

    # Session B attempts to delete Session A conversation
    del_leak = client.delete(f"/api/chat/conversations/{conv_id_a}?session_id={session_b}")
    assert del_leak.status_code == 404

