"""API Client for Streamlit to communicate with the FastAPI backend."""

import os
import requests
from typing import Dict, Any, List, Optional

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api")


class APIClient:
    """Handles all HTTP requests to the backend API."""

    @classmethod
    def check_health(cls) -> bool:
        try:
            res = requests.get(f"{BACKEND_URL.replace('/api', '')}/health", timeout=3)
            return res.status_code == 200
        except Exception:
            return False

    @classmethod
    def send_chat(
        cls,
        session_id: str,
        message: str,
        conversation_id: Optional[str] = None,
        modality: str = "text",
        media_url: Optional[str] = None,
        request_audio: bool = False,
        request_image: bool = False,
    ) -> Dict[str, Any]:
        url = f"{BACKEND_URL}/chat"
        payload = {
            "session_id": session_id,
            "conversation_id": conversation_id,
            "message": message,
            "modality": modality,
            "media_url": media_url,
            "request_audio": request_audio,
            "request_image": request_image,
        }
        res = requests.post(url, json=payload, timeout=60)
        res.raise_for_status()
        return res.json()

    @classmethod
    def list_conversations(cls, session_id: str) -> List[Dict[str, Any]]:
        url = f"{BACKEND_URL}/chat/conversations"
        res = requests.get(url, params={"session_id": session_id}, timeout=10)
        res.raise_for_status()
        return res.json()

    @classmethod
    def get_conversation(cls, conversation_id: str, session_id: str) -> Dict[str, Any]:
        url = f"{BACKEND_URL}/chat/conversations/{conversation_id}"
        res = requests.get(url, params={"session_id": session_id}, timeout=10)
        res.raise_for_status()
        return res.json()

    @classmethod
    def delete_conversation(cls, conversation_id: str, session_id: str) -> bool:
        url = f"{BACKEND_URL}/chat/conversations/{conversation_id}"
        res = requests.delete(url, params={"session_id": session_id}, timeout=10)
        return res.status_code == 200

    @classmethod
    def upload_document(cls, session_id: str, file_name: str, file_bytes: bytes) -> Dict[str, Any]:
        url = f"{BACKEND_URL}/documents/upload"
        files = {"file": (file_name, file_bytes)}
        data = {"session_id": session_id}
        res = requests.post(url, files=files, data=data, timeout=60)
        res.raise_for_status()
        return res.json()

    @classmethod
    def list_documents(cls, session_id: str) -> List[Dict[str, Any]]:
        url = f"{BACKEND_URL}/documents"
        res = requests.get(url, params={"session_id": session_id}, timeout=10)
        res.raise_for_status()
        return res.json().get("documents", [])

    @classmethod
    def delete_document(cls, document_id: str, session_id: str) -> bool:
        url = f"{BACKEND_URL}/documents/{document_id}"
        res = requests.delete(url, params={"session_id": session_id}, timeout=10)
        return res.status_code == 200

    @classmethod
    def upload_image(cls, file_name: str, file_bytes: bytes) -> Dict[str, Any]:
        url = f"{BACKEND_URL}/image/upload"
        files = {"file": (file_name, file_bytes)}
        res = requests.post(url, files=files, timeout=30)
        res.raise_for_status()
        return res.json()

    @classmethod
    def transcribe_audio(cls, file_name: str, file_bytes: bytes) -> Dict[str, Any]:
        url = f"{BACKEND_URL}/audio/transcribe"
        files = {"file": (file_name, file_bytes)}
        res = requests.post(url, files=files, timeout=30)
        res.raise_for_status()
        return res.json()

    @classmethod
    def submit_feedback(
        cls,
        conversation_id: str,
        message_id: Optional[str],
        rating: int,
        feedback_text: Optional[str] = None
    ) -> Dict[str, Any]:
        url = f"{BACKEND_URL}/feedback"
        payload = {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "rating": rating,
            "feedback_text": feedback_text,
        }
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        return res.json()

