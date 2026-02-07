# Friday AI Orchestrator - Production Container
# ===============================================
# Multi-stage build for the Friday orchestrator FastAPI server.
# DGX Spark: ARM64 (Grace CPU) + Blackwell GPU
#
# Usage:
#   docker build -t friday-orchestrator .
#   docker run -p 8000:8000 friday-orchestrator

FROM python:3.11-slim AS base

# System dependencies for audio, PDF processing, and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    poppler-utils \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (layer cached separately from code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY orchestrator/ ./orchestrator/
COPY memory/ ./memory/
COPY documents/ ./documents/
COPY mcp/ ./mcp/
COPY voice/ ./voice/
COPY db/ ./db/
COPY config/ ./config/
COPY prompts/ ./prompts/
COPY data/persona/ ./data/persona/

# Create directories for runtime data
RUN mkdir -p /app/data/documents /app/data/audio /app/logs /app/models

# Environment defaults
ENV FRIDAY_MODE=server \
    FRIDAY_HOST=0.0.0.0 \
    FRIDAY_PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()" || exit 1

CMD ["uvicorn", "orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
