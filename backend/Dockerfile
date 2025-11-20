FROM python:3.11-slim

WORKDIR /app/backend

# Install system dependencies (curl for healthcheck, ffmpeg for audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (in a separate layer for better caching)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source into /app/backend 
COPY . /app/backend

# For compiler code soon:
# COPY compiler/ /app/compiler/

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--app-dir", "/app"]