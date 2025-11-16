FROM python:3.11-slim

WORKDIR /app/backend

# Install system dependencies needed for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip

# Copy only necessary source files
COPY . /app/backend

# Remove unnecessary files to save space
RUN find /app/backend -name "*.pyc" -delete && \
    find /app/backend -name "__pycache__" -delete && \
    rm -rf /app/backend/.git* /app/backend/tests /app/backend/*.md

# Set environment variables with defaults that can be overridden
ARG ALLOWED_ORIGINS=http://localhost:3000,https://music-ai-assistant-frontend.onrender.com
ENV ALLOWED_ORIGINS=${ALLOWED_ORIGINS}

ARG RUNNER_INGEST_URL=https://music-ai-assistant-runner.onrender.com/eval
ENV RUNNER_INGEST_URL=${RUNNER_INGEST_URL}

# Optimize Python memory usage
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]