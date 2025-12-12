import os
import uuid
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv

import vertexai

# Conversation management
@dataclass
class Message:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    citations: Optional[List[str]] = None

@dataclass
class Conversation:
    id: str
    messages: List[Message]
    created_at: float
    updated_at: float

# In-memory conversation storage (for production, use Redis/database)
conversations: Dict[str, Conversation] = {}
MAX_CONVERSATIONS = 1000  # Limit to prevent memory issues
CONVERSATION_TIMEOUT = 3600 * 24  # 24 hours in seconds
try:
    from vertexai.generative_models import (
        GenerativeModel,
        GenerationConfig,
        Tool,
        Retrieval,
        VertexAISearch,
        VertexRagStore,
        RagResource,
    )
    _HAVE_TOOL_CLASSES = True
except Exception:
    try:
        from vertexai.preview.generative_models import (
            GenerativeModel,
            GenerationConfig,
            Tool,
            Retrieval,
            VertexAISearch,
            VertexRagStore,
            RagResource,
        )
        _HAVE_TOOL_CLASSES = True
    except Exception:
        from vertexai.preview.generative_models import (
            GenerativeModel,
            GenerationConfig,
        )
        _HAVE_TOOL_CLASSES = False
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.auth import default as google_auth_default
import requests


load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")
MODEL=gemini-2.5-flash-lite-preview-09-2025

# Retrieval configuration (choose one)
RAG_CORPUS = os.getenv("RAG_CORPUS")  # full resource name
DATA_STORE_ID = os.getenv("DATA_STORE_ID")  # ID only
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "5"))

vertexai.init(project=PROJECT_ID, location=LOCATION)


def build_retrieval_tools_json() -> list:
    if RAG_CORPUS:
        return [
            {
                "retrieval": {
                    "vertexRagStore": {
                        "ragResources": [{"ragCorpus": RAG_CORPUS}],
                        "similarityTopK": SIMILARITY_TOP_K,
                    }
                }
            }
        ]

    if DATA_STORE_ID:
        datastore_path = (
            f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/"
            f"default_collection/dataStores/{DATA_STORE_ID}"
        )
        return [
            {
                "retrieval": {
                    "vertexAiSearch": {"datastore": datastore_path}
                }
            }
        ]

    raise RuntimeError("Server not configured. Set RAG_CORPUS or DATA_STORE_ID in .env")


def extract_text_from_response(response) -> str:
    if not response:
        return ""
    # Prefer direct text property when available in newer SDKs
    direct_text = getattr(response, "text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()
    if not getattr(response, "candidates", None):
        return ""
    texts: List[str] = []
    for cand in response.candidates:
        parts = getattr(getattr(cand, "content", None), "parts", []) or []
        for p in parts:
            t = getattr(p, "text", None)
            if isinstance(t, str):
                texts.append(t)
    return "\n".join(texts).strip()


def extract_citations_from_response(response) -> List[str]:
    citations: List[str] = []
    if not response or not getattr(response, "candidates", None):
        return citations
    gm = getattr(response.candidates[0], "grounding_metadata", None)
    if not gm:
        return citations
    supports = getattr(gm, "grounding_supports", None) or []
    for s in supports:
        support = getattr(s, "support", None)
        if not support:
            continue
        chunk = getattr(support, "grounding_chunk", None)
        if not chunk:
            continue
        web = getattr(chunk, "web", None)
        file_info = getattr(chunk, "file", None)
        uri = None
        if web and getattr(web, "uri", None):
            uri = web.uri
        elif file_info and getattr(file_info, "file_path", None):
            uri = file_info.file_path
        if uri:
            citations.append(uri)
    return citations


def cleanup_old_conversations():
    """Remove conversations older than CONVERSATION_TIMEOUT"""
    current_time = time.time()
    to_remove = []

    for conv_id, conversation in conversations.items():
        if current_time - conversation.updated_at > CONVERSATION_TIMEOUT:
            to_remove.append(conv_id)

    for conv_id in to_remove:
        del conversations[conv_id]

    # If we still have too many conversations, remove oldest ones
    if len(conversations) > MAX_CONVERSATIONS:
        sorted_convs = sorted(conversations.values(), key=lambda x: x.updated_at)
        excess_count = len(conversations) - MAX_CONVERSATIONS
        for i in range(excess_count):
            del conversations[sorted_convs[i].id]


def get_or_create_conversation(conversation_id: Optional[str] = None) -> Conversation:
    """Get existing conversation or create new one"""
    cleanup_old_conversations()

    if conversation_id and conversation_id in conversations:
        conv = conversations[conversation_id]
        conv.updated_at = time.time()
        return conv

    # Create new conversation
    new_id = str(uuid.uuid4())
    conversation = Conversation(
        id=new_id,
        messages=[],
        created_at=time.time(),
        updated_at=time.time()
    )
    conversations[new_id] = conversation
    return conversation


def add_message_to_conversation(conversation: Conversation, role: str, content: str, citations: Optional[List[str]] = None):
    """Add a message to the conversation"""
    message = Message(
        role=role,
        content=content,
        timestamp=time.time(),
        citations=citations
    )
    conversation.messages.append(message)
    conversation.updated_at = time.time()


def format_conversation_history(conversation: Conversation, max_messages: int = 10) -> str:
    """Format conversation history for context in AI prompt"""
    recent_messages = conversation.messages[-max_messages:] if len(conversation.messages) > max_messages else conversation.messages

    formatted = []
    for msg in recent_messages:
        role_prefix = "User: " if msg.role == "user" else "Assistant: "
        formatted.append(f"{role_prefix}{msg.content}")

    return "\n\n".join(formatted)


app = Flask(__name__, static_folder="public", static_url_path="")


@app.get("/")
def index():
    return send_from_directory("public", "index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/conversation/<conversation_id>")
def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404

    conversation = conversations[conversation_id]
    return jsonify({
        "id": conversation.id,
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "citations": msg.citations
            }
            for msg in conversation.messages
        ],
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at
    })


@app.delete("/api/conversation/<conversation_id>")
def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    if conversation_id in conversations:
        del conversations[conversation_id]
        return jsonify({"success": True})
    return jsonify({"error": "Conversation not found"}), 404


@app.post("/api/conversation/new")
def new_conversation():
    """Create a new conversation"""
    conversation = get_or_create_conversation()  # This creates a new one
    return jsonify({"conversation_id": conversation.id})


@app.post("/api/ask")
def ask():
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        conversation_id = data.get("conversation_id")

        if not question:
            return jsonify({"error": "Missing question"}), 400
        if not PROJECT_ID:
            return jsonify({"error": "Set PROJECT_ID in .env"}), 500

        # Get or create conversation
        conversation = get_or_create_conversation(conversation_id)

        # Add user message to conversation
        add_message_to_conversation(conversation, "user", question)

        # Build the prompt with conversation history
        conversation_history = format_conversation_history(conversation)
        if conversation_history:
            enhanced_question = f"Previous conversation:\n{conversation_history}\n\nCurrent question: {question}"
        else:
            enhanced_question = question

        if _HAVE_TOOL_CLASSES:
            # Build typed Tool objects when SDK supports them
            tools_objects = []
            if RAG_CORPUS:
                tools_objects.append(
                    Tool.from_retrieval(
                        retrieval=Retrieval(
                            vertex_rag_store=VertexRagStore(
                                rag_resources=[RagResource(rag_corpus=RAG_CORPUS)],
                                similarity_top_k=SIMILARITY_TOP_K,
                            )
                        )
                    )
                )
            elif DATA_STORE_ID:
                datastore_path = (
                    f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/"
                    f"default_collection/dataStores/{DATA_STORE_ID}"
                )
                tools_objects.append(
                    Tool.from_retrieval(
                        retrieval=Retrieval(
                            vertex_ai_search=VertexAISearch(datastore=datastore_path)
                        )
                    )
                )
            model = GenerativeModel(MODEL, tools=tools_objects)
            response = model.generate_content(
                enhanced_question,
                generation_config=GenerationConfig(temperature=0.2),
            )
        else:
            # Fallback: direct REST call with JSON tools
            tools_json = build_retrieval_tools_json()
            endpoint = (
                f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/"
                f"{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL}:generateContent"
            )
            creds, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            creds.refresh(GoogleAuthRequest())
            payload = {
                "contents": [{"role": "user", "parts": [{"text": enhanced_question}]}],
                "tools": tools_json,
                "generationConfig": {"temperature": 0.2},
            }
            http_resp = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {creds.token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            if not http_resp.ok:
                return jsonify({"error": f"Vertex error: {http_resp.status_code} {http_resp.text}"}), 502
            response = http_resp.json()
            # For REST path, extract answer/citations directly from JSON
            texts = []
            for cand in response.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    t = part.get("text")
                    if isinstance(t, str):
                        texts.append(t)
            answer = "\n".join(texts).strip()
            citations = []
            gm = (response.get("candidates", [{}])[0]).get("groundingMetadata", {})
            for s in gm.get("groundingSupports", []):
                chunk = (s.get("support") or {}).get("groundingChunk") or {}
                uri = (chunk.get("web") or {}).get("uri") or (chunk.get("file") or {}).get("filePath")
                if uri:
                    citations.append(uri)
            # Add AI response to conversation
            add_message_to_conversation(conversation, "assistant", answer, citations)

            return jsonify({
                "answer": answer,
                "citations": citations,
                "conversation_id": conversation.id
            })

        answer = extract_text_from_response(response)
        citations = extract_citations_from_response(response)

        # Add AI response to conversation
        add_message_to_conversation(conversation, "assistant", answer, citations)

        return jsonify({
            "answer": answer,
            "citations": citations,
            "conversation_id": conversation.id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
