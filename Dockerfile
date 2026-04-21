FROM python:3.11-slim-bullseye
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Running the root app.py to host the UI on HuggingFace
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]