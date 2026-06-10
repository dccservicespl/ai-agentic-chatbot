# -------------------------------------------------------
# Stage: production image
# Base: python:3.13-slim (Debian-based, avoids Alpine
#       compilation failures with cryptography / psycopg2)
# -------------------------------------------------------
FROM python:3.13-slim

# -------------------------------------------------------
# System dependencies
#   gcc + libffi-dev  — cryptography>=46 (Rust/C build)
#   libpq-dev         — psycopg2-binary runtime libs
# Install and clean in one layer to keep image size down.
# -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------
# Poetry — installed into system Python.
# Virtualenv creation is disabled: packages go directly
# into the container's system Python, which is the only
# Python environment present.
# -------------------------------------------------------
RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false

WORKDIR /app

# -------------------------------------------------------
# Dependency install — cached layer.
# Copying lock files before source code means a source-
# only change does not invalidate this expensive layer.
# --only=main   excludes dev group (pytest etc.)
# --no-root     skips installing the project package;
#               PYTHONPATH handles imports instead.
# -------------------------------------------------------
COPY pyproject.toml poetry.lock ./
RUN poetry install --only=main --no-root --no-interaction --no-ansi

# -------------------------------------------------------
# Application source and supporting files
# -------------------------------------------------------
COPY src/ ./src/
COPY certs/ ./certs/
COPY config.example.yaml ./

# -------------------------------------------------------
# Runtime directories
# logging_config.py and transform_schema_to_text.py
# auto-create these, but pre-creating them here ensures
# correct ownership before the app process starts.
# -------------------------------------------------------
RUN mkdir -p logs temp

# -------------------------------------------------------
# Entrypoint script — validates env vars, waits for
# PostgreSQL, then execs CMD.
# -------------------------------------------------------
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# -------------------------------------------------------
# Non-root user — limits blast radius if app is
# compromised. chown ensures the app can write to
# logs/ and data/ at runtime.
# -------------------------------------------------------
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser \
    && chown -R appuser:appgroup /app

USER appuser

# -------------------------------------------------------
# Runtime environment
# -------------------------------------------------------
ENV PYTHONPATH=/app/src

EXPOSE 8000

# ENTRYPOINT runs validation + DB wait on every start.
# CMD is the default command passed to exec "$@" and can
# be overridden at docker run time without changing the
# entrypoint (e.g. add --reload for local dev).
ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "ai_agentic_chatbot.server:app", \
     "--host", "0.0.0.0", "--port", "8000"]