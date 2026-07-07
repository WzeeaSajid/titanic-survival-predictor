# ============================================================
# Stage 1 — Builder
# Install dependencies in a separate layer
# This keeps the final image clean and small
# ============================================================
FROM python:3.11-slim AS builder

# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr
# Critical for seeing logs in real time on AWS
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
# Docker caches this layer — if requirements.txt hasn't changed,
# pip install is skipped on rebuild. Saves minutes.
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ============================================================
# Stage 2 — Final Image
# Copy only what is needed to run the API
# ============================================================
FROM python:3.11-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user
# Never run production containers as root
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/       ./src/
COPY api/       ./api/
COPY models/    ./models/
COPY static/    ./static/

# Create directories the app writes to at runtime
RUN mkdir -p logs data/processed submission && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose the port FastAPI runs on
EXPOSE 8000

# Health check
# Docker and AWS will use this to know if the container is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Start the API
# --workers 2 is safe for t3.small (2 vCPU, 2GB RAM)
# increase on larger instances
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
