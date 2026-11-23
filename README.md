# 🕊️ Secure Multimodal AI Companion with RAG Architecture

A functional, modular, and secure personal AI companion featuring:
1. **Multimodal Input Processing**: Text, Image (PNG, JPG, WEBP), Audio (WAV, MP3, M4A), and Documents (PDF, TXT).
2. **Multimodal Output Generation**: Empathetic conversational text, optional Text-to-Speech (TTS) audio playback, and visual artwork generation.
3. **Session-Isolated RAG Pipeline**: Document chunking, vector embedding, similarity search with FAISS, and grounded source citations.
4. **Safety & Guardrails**: Dedicated crisis intervention layer (immediate 988 Crisis Lifeline and text line), prompt injection defenses, file upload sanitization, and non-medical boundaries.
5. **Modern Tech Stack**: FastAPI backend, Streamlit user interface, SQLAlchemy ORM with MySQL and Alembic migrations, and automated graceful SQLite fallback.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    User([User]) <--> UI[Streamlit Frontend :8501]
    UI <-->|REST API / JSON| API[FastAPI Backend :8000]

    subgraph "Backend Architecture"
        API --> Guardrails[Safety & Crisis Detection Layer]
        Guardrails -->|Crisis Detected| Hotlines[988 Lifeline Immediate Response]
        Guardrails -->|Safe Query| Orchestrator[Chat & Multimodal Orchestrator]

        Orchestrator --> RAG[RAG Retrieval Engine]
        RAG --> Embeddings[Embedding Provider]
        RAG --> FAISS[FAISS Vector Store<br/>Session Isolated]

        Orchestrator --> LLM[Configurable Model Providers<br/>OpenAI / Gemini / Local Fallback]
        Orchestrator --> Vision[Vision Provider<br/>OpenAI GPT-4o / Local PIL]
        Orchestrator --> AudioSTT[Speech-to-Text Provider<br/>Whisper / Local Parser]
        Orchestrator --> AudioTTS[Text-to-Speech Provider<br/>gTTS / OpenAI TTS]
        Orchestrator --> ImageGen[Image Gen Provider<br/>DALL-E 3 / Local Generator]

        Orchestrator --> DB[(MySQL / SQLAlchemy ORM)]
    end

    subgraph "Storage Layer"
        DB --- Tables[conversations<br/>messages<br/>documents<br/>document_chunks<br/>feedback]
        FAISS --- IndexFile[faiss.index + metadata.json]
        Orchestrator --- DiskStorage[storage/uploads<br/>storage/generated]
    end
```

---

## 📦 Database Schema

The database design uses SQLAlchemy ORM with foreign keys, indexes, and cascading deletion policies:

- **`conversations`**:
  - `id` (VARCHAR 36, PK UUID)
  - `session_id` (VARCHAR 64, Index)
  - `title` (VARCHAR 255)
  - `created_at`, `updated_at` (DATETIME)
  - *Cascades to messages and feedback on deletion.*

- **`messages`**:
  - `id` (VARCHAR 36, PK UUID)
  - `conversation_id` (VARCHAR 36, FK `conversations.id` ON DELETE CASCADE, Index)
  - `role` (VARCHAR 20: user / assistant / system)
  - `content` (TEXT)
  - `modality` (VARCHAR 30: text / image / audio / multimodal)
  - `raw_metadata` (TEXT / JSON: citations, audio_url, image_url, crisis flags)
  - `created_at` (DATETIME)

- **`documents`**:
  - `id` (VARCHAR 36, PK UUID)
  - `session_id` (VARCHAR 64, Index)
  - `filename` (VARCHAR 255)
  - `file_type` (VARCHAR 50: pdf / txt)
  - `file_size` (INT)
  - `storage_path` (VARCHAR 500)
  - `processing_status` (VARCHAR 50: pending / processed / failed)
  - `created_at` (DATETIME)
  - *Cascades to document_chunks on deletion.*

- **`document_chunks`**:
  - `id` (VARCHAR 36, PK UUID)
  - `document_id` (VARCHAR 36, FK `documents.id` ON DELETE CASCADE, Index)
  - `chunk_index` (INT)
  - `chunk_text` (TEXT)
  - `embedding_reference` (VARCHAR 100)
  - `created_at` (DATETIME)

- **`feedback`**:
  - `id` (VARCHAR 36, PK UUID)
  - `conversation_id` (VARCHAR 36, FK `conversations.id` ON DELETE CASCADE)
  - `message_id` (VARCHAR 36, nullable)
  - `rating` (INT: -1 to 5)
  - `feedback_text` (TEXT)
  - `created_at` (DATETIME)

---

## 🚀 Prerequisites

- **Python**: 3.10 to 3.14
- **MySQL**: MySQL 8.0+ (Optional: If MySQL is not running locally, the system automatically and gracefully falls back to local SQLite at `./multimodal_ai.db`).
- **Git** (optional)

---

## 🛠️ Installation

1. **Clone or Navigate to the Repository**:
   ```bash
   cd multimodal_ai_companion
   ```

2. **Create and Activate a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration & Environment Variables

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Key environment options in `.env`:
```ini
# Application
PROJECT_NAME="Secure Multimodal AI Companion"
ENVIRONMENT="development"
DEBUG=True

# Database (MySQL primary; automatic SQLite fallback if unreachable)
DATABASE_URL="mysql+pymysql://root:password@localhost:3306/multimodal_ai"

# Providers
LLM_PROVIDER="fallback"          # "fallback" (offline local) | "openai" | "gemini"
EMBEDDING_PROVIDER="local"       # "local" (deterministic FAISS) | "openai"
VISION_PROVIDER="fallback"       # "fallback" (local PIL) | "openai"
STT_PROVIDER="fallback"          # "fallback" (local audio parser) | "openai"
TTS_PROVIDER="gtts"              # "gtts" (Google TTS offline/free) | "openai" | "fallback"
IMAGE_GEN_PROVIDER="fallback"    # "fallback" (local graphic synthesis) | "openai"

# API Keys (Optional - only required when using "openai" or "gemini" providers)
OPENAI_API_KEY=""
OPENAI_MODEL="gpt-4o-mini"
GEMINI_API_KEY=""
GEMINI_MODEL="gemini-1.5-flash"

# RAG Parameters
SIMILARITY_THRESHOLD=0.35
MAX_RETRIEVAL_CHUNKS=4
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Upload Security Limits
MAX_UPLOAD_SIZE_BYTES=20971520   # 20 MB

# Frontend Connection
BACKEND_URL="http://127.0.0.1:8000/api"
```

---

## 🗄️ Database Initialization & Migrations

### Using Alembic Migrations:
Run database migrations using Alembic:
```bash
alembic upgrade head
```

The database schema is also auto-initialized by SQLAlchemy on FastAPI startup if tables do not exist.

---

## 🏃 Running the Application

### 1. Start the FastAPI Backend
In your terminal:
```bash
uvicorn multimodal_ai_companion.backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### 2. Start the Streamlit Frontend
In a second terminal:
```bash
streamlit run multimodal_ai_companion/frontend/app.py
```
- Open your browser to [http://localhost:8501](http://localhost:8501)

---

## 🧪 Automated Testing

The project includes an automated test suite covering all 20 required scenarios:

```bash
python -m pytest multimodal_ai_companion/tests -v
```

### Verified Scenarios:
1. `test_scenario_01_normal_text_conversation`: Basic text conversation turn.
2. `test_scenario_02_follow_up_question_using_context`: Multi-turn context preservation.
3. `test_scenario_03_pdf_upload_and_rag_qa`: PDF extraction with `pypdf`, FAISS indexing, and retrieval.
4. `test_scenario_04_txt_upload_and_rag_qa`: TXT ingestion and grounded question answering.
5. `test_scenario_05_image_upload_and_understanding`: Image inspection (dimensions, color, orientation, content).
6. `test_scenario_06_audio_upload_and_transcription`: Audio file transcription (WAV, MP3).
7. `test_scenario_07_text_to_speech_generation`: Speech synthesis (gTTS / WAV generator) and audio playback stream.
8. `test_scenario_08_optional_image_generation`: Prompt-based visual image synthesis.
9. `test_scenario_09_question_with_no_relevant_rag_context`: Graceful limitation message with zero fabricated citations.
10. `test_scenario_10_malicious_instructions_in_uploaded_document`: Embedded prompt injections in documents treated strictly as untrusted text.
11. `test_scenario_11_prompt_injection_attempts`: Jailbreak & system override attempts defused.
12. `test_scenario_12_unsupported_file_upload`: Rejection of unsupported formats (.exe, .sh) with 400 Bad Request.
13. `test_scenario_13_oversized_file_upload`: Rejection of files exceeding 20MB limit.
14. `test_scenario_14_database_connection_fallback`: Graceful fallback to SQLite when MySQL daemon is offline.
15. `test_scenario_15_safe_exception_handling_on_provider_failure`: 500 Internal Server Error without leaking internal stack traces.
16. `test_scenario_16_conversation_deletion`: Cascading deletion of conversation threads and messages.
17. `test_scenario_17_cross_session_data_isolation`: Strict verification that Session A cannot access or delete Session B data.
18. `test_scenario_18_emotional_distress_conversation`: Empathetic non-judgmental tone with medical disclaimer.
19. `test_scenario_19_self_harm_or_crisis_scenario`: Rule-based crisis detection layer returning immediate 988 Crisis Lifeline.
20. `test_scenario_20_missing_or_invalid_api_key`: Zero-API-key graceful operation using built-in provider fallback.

---

## 🔒 Security & Privacy Considerations

- **Untrusted Document Context**: RAG chunks are encapsulated in `<document_reference>` tags to prevent delimiter confusion and context override.
- **Dedicated Crisis Layer**: Crisis statements are evaluated via dedicated pattern analysis before LLM invocation, ensuring safety hotlines (988 Lifeline, Crisis Text Line 741741) are delivered immediately.
- **Session Isolation**: Documents and FAISS vector indices are tagged by `session_id`. Queries strictly filter out chunks from other sessions.
- **Path Traversal Defenses**: Filenames are sanitized via `re.sub(r"[^a-zA-Z0-9_.-]", "_")` and stored under unique UUID prefixes on disk.
- **Zero API Key Leakage**: Keys remain exclusively in `.env` and are never logged or returned in API responses.

---

## ⚠️ Current Limitations (MVP Scope)

1. **Authentication Excluded**: This MVP is designed for trusted local environments. User login, JWTs, and RBAC should be added prior to public multi-tenant deployment.
2. **FAISS Scalability**: The MVP uses local file-backed FAISS indexes. Production high-scale deployments can swap the retrieval layer with Qdrant, Milvus, or pgvector using the existing `FAISSVectorStore` interface.
3. **Audio File Formats**: STT and TTS currently support WAV, MP3, and M4A.

