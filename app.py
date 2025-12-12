import os
import uuid
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Tool,
    VertexAISearch,
)

# 1. تحميل متغيرات البيئة
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")
# تصحيح: إضافة علامات التنصيص واستخدام نموذج موجود فعلياً
MODEL = os.getenv("MODEL", "gemini-1.5-flash") 
DATA_STORE_ID = os.getenv("DATA_STORE_ID")

# 2. تهيئة Vertex AI
if PROJECT_ID:
    vertexai.init(project=PROJECT_ID, location=LOCATION)

# 3. إدارة المحادثات (بسيطة للتجربة)
@dataclass
class Conversation:
    id: str
    messages: List[dict]
    updated_at: float

conversations: Dict[str, Conversation] = {}

app = Flask(__name__, static_folder="public", static_url_path="")

# --- الدوال المساعدة ---

def get_or_create_conversation(conv_id: Optional[str]) -> Conversation:
    if conv_id in conversations:
        return conversations[conv_id]
    
    new_id = str(uuid.uuid4())
    conv = Conversation(id=new_id, messages=[], updated_at=time.time())
    conversations[new_id] = conv
    return conv

# --- المسارات (Routes) ---

@app.get("/")
def index():
    # تأكد من وجود مجلد public وملف index.html بداخله
    try:
        return send_from_directory("public", "index.html")
    except:
        return "<h1>تطبيق RAG يعمل بنجاح</h1><p>ملاحظة: ملف index.html مفقود في مجلد public.</p>", 200

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "project_id": PROJECT_ID})

@app.post("/api/ask")
def ask():
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        conv_id = data.get("conversation_id")

        if not question:
            return jsonify({"error": "يرجى إدخال سؤال"}), 400
        
        if not DATA_STORE_ID:
            return jsonify({"error": "DATA_STORE_ID غير مضبوط في الإعدادات"}), 500

        # جلب المحادثة
        conversation = get_or_create_conversation(conv_id)

        # تجهيز أداة البحث (Vertex AI Search)
        datastore_path = f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/dataStores/{DATA_STORE_ID}"
        tools = [Tool.from_retrieval(
            retrieval=vertexai.generative_models.Retrieval(
                vertex_ai_search=VertexAISearch(datastore=datastore_path)
            )
        )]

        # إنشاء النموذج واستدعاؤه
        model = GenerativeModel(MODEL, tools=tools)
        
        # إضافة سياق المحادثة (اختياري لبساطة الرد)
        response = model.generate_content(
            question,
            generation_config=GenerationConfig(temperature=0.2)
        )

        # استخراج النص والمصادر
        answer = response.text if response.text else "لم أتمكن من العثور على إجابة."
        
        # استخراج الروابط (Citations) إذا وجدت
        citations = []
        try:
            if response.candidates[0].grounding_metadata.grounding_chunks:
                for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
                    uri = chunk.web.uri if chunk.web else chunk.file.file_path
                    if uri and uri not in citations:
                        citations.append(uri)
        except:
            pass

        # حفظ الرسائل
        conversation.messages.append({"role": "user", "content": question})
        conversation.messages.append({"role": "assistant", "content": answer})
        conversation.updated_at = time.time()

        return jsonify({
            "answer": answer,
            "citations": citations,
            "conversation_id": conversation.id
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"حدث خطأ داخلي: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
