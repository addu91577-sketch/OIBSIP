# ============================================================================
# Google Play Store Analytics & Smart-Keyboard REST API Container
# ============================================================================
FROM python:3.11-slim AS base

# System environment configurations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code and verified datasets
COPY src/ ./src/
COPY Data/ ./Data/
COPY reports/ ./reports/
COPY Notebooks/ ./Notebooks/
COPY tests/ ./tests/

# Expose microservice ports: 8000 (FastAPI REST API), 8501 (Streamlit Dashboard)
EXPOSE 8000 8501

# Health check endpoint probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default runtime entrypoint: Uvicorn ASGI server
CMD ["uvicorn", "src.production_api:app", "--host", "0.0.0.0", "--port", "8000"]
