"""Streamlit Frontend Application.

A modern, responsive conversational UI for the Secure Multimodal AI Companion.
Supports text, document RAG, image understanding, audio STT/TTS, and emotional safety.
"""

import os
import uuid
import streamlit as st
from multimodal_ai_companion.frontend.utils.api_client import APIClient, BACKEND_URL

# Page configuration
st.set_page_config(
    page_title="Secure Multimodal AI Companion",
    page_icon="🕊️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal CSS for clean typography and spacing
st.markdown(
    """
    <style>
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .crisis-card {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        color: #991B1B;
    }
    .rag-badge {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 0.15rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.6rem;
        margin-top: 0.4rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 1. Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "request_audio" not in st.session_state:
    st.session_state.request_audio = False
if "request_image" not in st.session_state:
    st.session_state.request_image = False


def reset_session():
    """Generates a new session and resets state."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.current_conversation_id = None
    st.session_state.messages = []
    st.rerun()


def switch_conversation(conv_id: str):
    """Loads a specific conversation."""
    st.session_state.current_conversation_id = conv_id
    try:
        data = APIClient.get_conversation(conv_id, st.session_state.session_id)
        st.session_state.messages = []
        for m in data.get("messages", []):
            st.session_state.messages.append({
                "role": m["role"],
                "content": m["content"],
                "modality": m.get("modality", "text"),
                "raw_metadata": m.get("raw_metadata"),
            })
    except Exception as e:
        st.error(f"Error loading conversation: {e}")


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🕊️ AI Companion")
    st.caption("Secure • Empathetic • Multimodal RAG")

    st.markdown(f"**Session:** `{st.session_state.session_id[:8]}...`")
    if st.button("🔄 New Session", use_container_width=True):
        reset_session()

    st.divider()

    # Conversation Management
    col_new_conv, col_clear = st.columns([1, 1])
    with col_new_conv:
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.current_conversation_id = None
            st.session_state.messages = []
            st.rerun()

    # Load Conversation Threads
    try:
        conversations = APIClient.list_conversations(st.session_state.session_id)
        if conversations:
            st.markdown("**Recent Conversations:**")
            for conv in conversations[:8]:
                title = conv.get("title", "Untitled")
                cid = conv.get("id")
                active_prefix = "👉 " if cid == st.session_state.current_conversation_id else "💬 "
                c1, c2 = st.columns([4, 1])
                with c1:
                    if st.button(f"{active_prefix}{title[:22]}", key=f"conv_{cid}", use_container_width=True):
                        switch_conversation(cid)
                        st.rerun()
                with c2:
                    if st.button("🗑️", key=f"del_{cid}"):
                        APIClient.delete_conversation(cid, st.session_state.session_id)
                        if st.session_state.current_conversation_id == cid:
                            st.session_state.current_conversation_id = None
                            st.session_state.messages = []
                        st.rerun()
    except Exception:
        st.caption("⚠️ Could not fetch conversation history from backend.")

    st.divider()

    # Document Management (RAG)
    st.markdown("### 📄 RAG Documents")
    uploaded_doc = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        help="Documents uploaded here are chunked and indexed into FAISS for this session only.",
    )

    if uploaded_doc is not None:
        if st.button("📥 Index Document", use_container_width=True):
            with st.spinner("Extracting text and generating FAISS embeddings..."):
                try:
                    res = APIClient.upload_document(
                        session_id=st.session_state.session_id,
                        file_name=uploaded_doc.name,
                        file_bytes=uploaded_doc.getvalue(),
                    )
                    st.success(f"Indexed: {res.get('chunks_created')} chunks created!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    # List Indexed Documents
    try:
        docs = APIClient.list_documents(st.session_state.session_id)
        if docs:
            st.markdown(f"**Session Documents ({len(docs)}):**")
            for d in docs:
                dc1, dc2 = st.columns([4, 1])
                with dc1:
                    st.caption(f"📑 {d['filename']} ({d['chunk_count']} chunks)")
                with dc2:
                    if st.button("❌", key=f"deldoc_{d['id']}"):
                        APIClient.delete_document(d["id"], st.session_state.session_id)
                        st.rerun()
        else:
            st.caption("No documents indexed yet.")
    except Exception:
        pass

    st.divider()

    # Settings
    st.markdown("### ⚙️ Modality Settings")
    st.session_state.request_audio = st.checkbox(
        "Generate Voice Response (TTS)",
        value=st.session_state.request_audio,
        help="Synthesizes speech audio for every assistant response.",
    )
    st.session_state.request_image = st.checkbox(
        "Request Visual Artwork",
        value=st.session_state.request_image,
        help="Instructs companion to generate an image accompanying the response.",
    )

    st.caption("🔒 Privacy: Isolated session vector store & DB.")


# --- MAIN CHAT INTERFACE ---
st.markdown('<div class="main-title">🕊️ Secure Multimodal AI Companion</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Empathetic Personal Support • Multimodal Input & Output • Grounded RAG Knowledge</div>',
    unsafe_allow_html=True,
)

# Backend health check banner
if not APIClient.check_health():
    st.warning("⚠️ Backend API server appears unreachable at " + BACKEND_URL + ". Ensure FastAPI is running on port 8000.")

# Render Chat History
if not st.session_state.messages:
    # Empty state welcome banner
    st.info(
        "👋 Welcome! I am here to support you. You can talk freely about your thoughts or emotions, "
        "upload documents for question-answering, inspect images, or transcribe voice notes."
    )
else:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        avatar = "👤" if role == "user" else "🕊️"

        with st.chat_message(role, avatar=avatar):
            # Check if this message was a crisis response
            if role == "assistant" and "988" in content and "Crisis Lifeline" in content:
                st.markdown(
                    """
                    <div class="crisis-card">
                        <strong>🚨 Crisis Support Resources Available</strong><br>
                        Please know that you are not alone and help is available right now.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Check if RAG citations are present
            citations = msg.get("citations", [])
            if citations:
                st.markdown('<span class="rag-badge">🔍 Grounded in Uploaded Documents</span>', unsafe_allow_html=True)

            st.markdown(content)

            # Citations dropdown
            if citations:
                with st.expander("📚 View Document Citations"):
                    for c in citations:
                        st.markdown(
                            f"""
                            <div class="source-box">
                                <strong>Source:</strong> {c.get('filename')} (Chunk #{c.get('chunk_index')})<br>
                                <em>"{c.get('text_snippet')}"</em><br>
                                <small>Match Score: {c.get('score')}</small>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Display Media outputs
            if msg.get("audio_url"):
                audio_full_url = f"{BACKEND_URL.replace('/api', '')}{msg['audio_url']}"
                st.audio(audio_full_url)

            if msg.get("image_url"):
                img_full_url = f"{BACKEND_URL.replace('/api', '')}{msg['image_url']}"
                st.image(img_full_url, caption="Generated Visual Artwork", use_container_width=True)


# --- ATTACHMENT EXPANDER FOR MULTIMODAL INPUTS ---
attached_media_path = None
input_modality = "text"

with st.expander("📎 Attach Multimodal Media (Optional Image / Audio)"):
    tab_img, tab_aud = st.tabs(["🖼️ Image Understanding", "🎙️ Audio Note (STT)"])

    with tab_img:
        uploaded_image = st.file_uploader("Upload Image (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"], key="chat_img_uploader")
        if uploaded_image:
            st.image(uploaded_image, caption="Preview Attached Image", width=250)
            if st.button("Process Image Attachment", key="btn_img_attach"):
                with st.spinner("Uploading image..."):
                    img_res = APIClient.upload_image(uploaded_image.name, uploaded_image.getvalue())
                    attached_media_path = img_res.get("image_path")
                    input_modality = "image"
                    st.success("Image attached! Type your question below.")

    with tab_aud:
        uploaded_audio = st.file_uploader("Upload Audio (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"], key="chat_aud_uploader")
        if uploaded_audio:
            if st.button("Transcribe Voice Note", key="btn_aud_transcribe"):
                with st.spinner("Transcribing audio..."):
                    aud_res = APIClient.transcribe_audio(uploaded_audio.name, uploaded_audio.getvalue())
                    transcribed_text = aud_res.get("text", "")
                    st.session_state["transcribed_draft"] = transcribed_text
                    st.success(f"Transcribed: \"{transcribed_text}\"")

# Prefill prompt from speech-to-text if available
default_prompt = st.session_state.pop("transcribed_draft", "")

# Chat text input
user_prompt = st.chat_input("Message your companion... (e.g. 'How can I cope with stress today?' or ask about your notes)")
if default_prompt and not user_prompt:
    user_prompt = default_prompt

if user_prompt:
    # 1. Append user message to UI immediately
    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt,
        "modality": input_modality,
    })
    st.rerun()

# If the last message is from user and needs assistant reply
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_msg = st.session_state.messages[-1]

    with st.chat_message("assistant", avatar="🕊️"):
        with st.spinner("Reflecting thoughtfully..."):
            try:
                response = APIClient.send_chat(
                    session_id=st.session_state.session_id,
                    message=last_user_msg["content"],
                    conversation_id=st.session_state.current_conversation_id,
                    modality=last_user_msg.get("modality", "text"),
                    media_url=attached_media_path,
                    request_audio=st.session_state.request_audio,
                    request_image=st.session_state.request_image,
                )

                # Update current conversation id
                st.session_state.current_conversation_id = response.get("conversation_id")

                # Parse assistant response
                assistant_record = {
                    "role": "assistant",
                    "content": response.get("content", ""),
                    "modality": response.get("modality", "text"),
                    "citations": response.get("citations", []),
                    "audio_url": response.get("audio_url"),
                    "image_url": response.get("image_url"),
                    "is_crisis": response.get("is_crisis", False),
                }

                st.session_state.messages.append(assistant_record)
                st.rerun()

            except Exception as e:
                st.error(f"Failed to generate response: {e}")

