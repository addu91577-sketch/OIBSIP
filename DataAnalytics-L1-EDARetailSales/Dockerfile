# Multi-Stage Production Container for Retail Sales Analytics & Machine Learning
FROM python:3.11-slim as base

# Set environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code, datasets, and presentation
COPY src/ ./src/
COPY Data/ ./Data/
COPY presentation/ ./presentation/
COPY tests/ ./tests/
COPY pytest.ini .
COPY README.md .

# Create non-root runtime user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports: 8000 for FastAPI REST API, 8501 for Streamlit Dashboard
EXPOSE 8000 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run FastAPI REST API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
