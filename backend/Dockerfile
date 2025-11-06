FROM python:3.11-slim

WORKDIR /app/backend

# Install deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source into /app/backend 
COPY . /app/backend

# For compiler code soon:
# COPY compiler/ /app/compiler/

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--app-dir", "/app"]