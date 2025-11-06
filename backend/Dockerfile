FROM python:3.11-slim

WORKDIR /app/backend

# Install deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source into /app/backend
COPY . /app/backend

# For compiler code soon:
# COPY compiler/ /app/compiler/

# Set environment variables with defaults that can be overridden
ARG ALLOWED_ORIGINS=http://localhost:3000,https://music-ai-assistant-frontend.onrender.com
ENV ALLOWED_ORIGINS=${ALLOWED_ORIGINS}

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]