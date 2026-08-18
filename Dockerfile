FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# System packages required by scientific Python wheels/runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install production dependencies first for better Docker layer caching.
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source and production artifacts.
COPY app.py ./
COPY src ./src
COPY static ./static
COPY models ./models

# Runtime data is intentionally not copied into the image.
# The application expects these files under /app/data:
#   data/raw/train_transaction.csv
#   data/raw/train_identity.csv
#   data/investigation_history.csv
#   data/risk_assessments.json
# Mount/provide the data directory at runtime.
RUN mkdir -p /app/data/raw

# Run as an unprivileged user.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
