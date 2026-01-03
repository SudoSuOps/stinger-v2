# =============================================================================
# STINGER V2 — Intelligent Medical AI Gateway
# stinger.swarmbee.eth
# =============================================================================

FROM python:3.11-slim

# Labels
LABEL maintainer="TrustCat <dev@trustcat.ai>"
LABEL description="Stinger V2 - Intelligent Medical AI Gateway"
LABEL version="2.0.0"
LABEL org.opencontainers.image.source="https://github.com/swarmhealth/stinger-v2"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user
RUN groupadd -r stinger && useradd -r -g stinger stinger

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip setuptools wheel && \
    pip install ".[full]"

# Copy application
COPY stinger/ ./stinger/
COPY tests/ ./tests/

# Create directories
RUN mkdir -p /opt/stinger/outputs /opt/stinger/models /opt/stinger/prompts && \
    chown -R stinger:stinger /opt/stinger

# Switch to non-root user
USER stinger

# Expose port
EXPOSE 8100

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8100/ || exit 1

# Default command
CMD ["uvicorn", "stinger.main:app", "--host", "0.0.0.0", "--port", "8100"]
