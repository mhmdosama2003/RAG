# استخدام صورة بايثون خفيفة لتوفير المساحة
FROM python:3.11-slim

# تعيين مجلد العمل
WORKDIR /app

# نسخ ملف المتطلبات وتثبيتها أولاً (هذا الأمر هو الذي يحل مشكلة gunicorn)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع (app.py و public)
COPY . .

# أمر تشغيل التطبيق باستخدام gunicorn على المنفذ المتغير $PORT
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
