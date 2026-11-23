"""Tests for Multimodal Inputs and Outputs (Scenarios 5, 6, 7, 8)."""

import io
import uuid
from PIL import Image
from fastapi.testclient import TestClient


def create_sample_png_bytes(width: int = 100, height: int = 100) -> bytes:
    """Helper to generate a clean PNG in-memory."""
    img = Image.new("RGB", (width, height), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_sample_wav_bytes() -> bytes:
    """Helper to generate a valid minimal WAV in-memory."""
    import wave
    import struct

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        for i in range(1600):
            wf.writeframesraw(struct.pack("<h", 0))
    return buf.getvalue()


def test_scenario_05_image_upload_and_understanding(client: TestClient):
    """Scenario 5: Upload image, inspect properties, and query image in chat."""
    session_id = str(uuid.uuid4())
    img_bytes = create_sample_png_bytes(200, 150)

    # 1. Upload image
    upload_res = client.post(
        "/api/image/upload",
        files={"file": ("chart.png", img_bytes, "image/png")}
    )
    assert upload_res.status_code == 200
    img_data = upload_res.json()
    assert "image_path" in img_data

    # 2. Inspect image directly
    analyze_res = client.post("/api/image/analyze", json={
        "image_path": img_data["image_path"],
        "prompt": "What are the dimensions and orientation of this chart?"
    })
    assert analyze_res.status_code == 200
    analysis = analyze_res.json()
    assert analysis["width"] == 200
    assert analysis["height"] == 150

    # 3. Use in chat turn
    chat_res = client.post("/api/chat", json={
        "session_id": session_id,
        "message": "Can you inspect my attached diagram?",
        "modality": "image",
        "media_url": img_data["image_path"]
    })
    assert chat_res.status_code == 200
    assert len(chat_res.json()["content"]) > 0


def test_scenario_06_audio_upload_and_transcription(client: TestClient):
    """Scenario 6: Upload audio recording and convert speech to text."""
    wav_bytes = create_sample_wav_bytes()

    res = client.post(
        "/api/audio/transcribe",
        files={"file": ("voice_memo.wav", wav_bytes, "audio/wav")}
    )
    assert res.status_code == 200
    data = res.json()
    assert "text" in data
    assert len(data["text"]) > 0
    assert data["language"] == "en"


def test_scenario_07_text_to_speech_generation(client: TestClient):
    """Scenario 7: Request voice/audio synthesis for chat response."""
    session_id = str(uuid.uuid4())

    # Direct audio generation endpoint
    gen_res = client.post("/api/audio/generate", json={
        "text": "Hello, this is a spoken audio response from your companion.",
        "session_id": session_id
    })
    assert gen_res.status_code == 200
    gen_data = gen_res.json()
    assert "audio_url" in gen_data

    # Verify audio file streaming
    stream_res = client.get(gen_data["audio_url"])
    assert stream_res.status_code == 200
    assert len(stream_res.content) > 0


def test_scenario_08_optional_image_generation(client: TestClient):
    """Scenario 8: Explicitly request image generation and verify image synthesis output."""
    session_id = str(uuid.uuid4())

    chat_res = client.post("/api/chat", json={
        "session_id": session_id,
        "message": "Please generate an image of a tranquil mountain lake at sunrise.",
        "request_image": True
    })
    assert chat_res.status_code == 200
    data = chat_res.json()
    assert data["image_url"] is not None
    assert "/api/image/file/" in data["image_url"]

    # Verify generated image can be fetched
    img_fetch = client.get(data["image_url"])
    assert img_fetch.status_code == 200
    assert img_fetch.headers["content-type"].startswith("image/")
    assert len(img_fetch.content) > 0

