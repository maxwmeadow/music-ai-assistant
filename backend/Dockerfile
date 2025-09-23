FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source into /app/backend 
COPY backend/ /app/backend/
# For compiler code soon:
# COPY compiler/ /app/compiler/

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
