import os
import uuid
import time
from typing import List, Dict, Optional
from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Tool, VertexAISearch

# 1. إعداد المسارات المطلقة لضمان الوصول لمجلد public داخل الحاوية
load_dotenv()
basedir = os.path.abspath(os.path.dirname(__file__))

# 2. قراءة متغيرات البيئة (سيتم جلبها من إعدادات Cloud Run)
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")
# تصحيح: استخدام نموذج مستقر ومعروف لتجنب انهيار التطبيق
MODEL = os.getenv("MODEL", "gemini-1.5-flash") 
DATA_STORE_ID = os.getenv("DATA_STORE_ID")

# 3. تهيئة Vertex AI
if PROJECT_ID:
    vertexai.init(project=PROJECT_ID, location=LOCATION)

# 4. إعداد تطبيق Flask وتحديد مسار مجلد السكون بشكل مطلق
app = Flask(__name__, 
            static_folder=os.path.join(basedir, 'public'), 
            static_url_path='')

@app.route("/")
def index():
    """تقديم ملف الواجهة الأمامية index.html"""
    try:
        # التأكد من وجود الملف قبل محاولة تقديمه لتجنب خطأ 500 الصامت
        if not os.path.exists(os.path.join(app.static_folder, "index.html")):
            return f"<h1>التطبيق يعمل!</h1><p>تنبيه: لم يتم العثور على index.html في المسار: {app.static_folder}</p>", 200
        return send_from_directory(app.static_folder, "index.html")
    except Exception as e:
        print(f"Error serving index.html: {e}")
        return f"خطأ داخلي في الخادم: {str(e)}", 500

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "project_id": PROJECT_ID})

@app.route("/api/ask", methods=["POST"])
def ask():
    """معالجة أسئلة المستخدم باستخدام تقنية RAG"""
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "يرجى إرسال سؤال صحيح"}), 400

        if not DATA_STORE_ID:
            return jsonify({"error": "إعدادات DATA_STORE_ID مفقودة"}), 500

        # إعداد مسار البحث في مخزن البيانات
        datastore_path = f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/dataStores/{DATA_STORE_ID}"
        
        # تجهيز أدوات البحث (Vertex AI Search)
        tools = [Tool.from_retrieval(
            vertexai.generative_models.Retrieval(
                vertex_ai_search=VertexAISearch(datastore=datastore_path)
            )
        )]

        # استدعاء نموذج Gemini
        model = GenerativeModel(MODEL)
        response = model.generate_content(
            question,
            tools=tools,
            generation_config=GenerationConfig(temperature=0.2)
        )

        # استخراج الإجابة
        answer = response.text if response.text else "عذراً، لم أتمكن من العثور على إجابة في الوثائق المتاحة."

        return jsonify({
            "answer": answer,
            "conversation_id": str(uuid.uuid4())
        })
    except Exception as e:
        print(f"Internal Error in /api/ask: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # استخدام المنفذ المحدد من قبل Google Cloud
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
