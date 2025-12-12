import os
import uuid
import time
from typing import List, Dict, Optional
from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Tool, VertexAISearch

# 1. تحميل الإعدادات
load_dotenv()
basedir = os.path.abspath(os.path.dirname(__file__)) # المسار الحالي للمشروع

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")
# تصحيح: استخدام نموذج مستقر ومعروف
MODEL = os.getenv("MODEL", "gemini-1.5-flash") 
DATA_STORE_ID = os.getenv("DATA_STORE_ID")

# 2. تهيئة Vertex AI
if PROJECT_ID:
    vertexai.init(project=PROJECT_ID, location=LOCATION)

# 3. إعداد تطبيق Flask مع تحديد مسار مجلد السكون
# نستخدم os.path.join لضمان المسار المطلق لمجلد public
app = Flask(__name__, 
            static_folder=os.path.join(basedir, 'public'), 
            static_url_path='')

@app.route("/")
def index():
    """تقديم ملف index.html من مجلد public"""
    try:
        # البحث عن الملف في المسار المطلق
        return send_from_directory(app.static_folder, "index.html")
    except Exception as e:
        print(f"Error serving index.html: {e}")
        return f"<h1>التطبيق يعمل</h1><p>ولكن لم يتم العثور على واجهة المستخدم في: {app.static_folder}</p>", 200

@app.route("/api/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "No question provided"}), 400

        datastore_path = f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/dataStores/{DATA_STORE_ID}"
        tools = [Tool.from_retrieval(
            vertexai.generative_models.Retrieval(
                vertex_ai_search=VertexAISearch(datastore=datastore_path)
            )
        )]

        model = GenerativeModel(MODEL)
        response = model.generate_content(
            question,
            tools=tools,
            generation_config=GenerationConfig(temperature=0.2)
        )

        return jsonify({
            "answer": response.text,
            "conversation_id": str(uuid.uuid4())
        })
    except Exception as e:
        print(f"Internal Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
