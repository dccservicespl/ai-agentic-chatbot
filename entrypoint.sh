#!/bin/sh
set -e

# -------------------------------------------------------
# Helper: fail fast with a clear message if a required
# environment variable is missing or empty.
# -------------------------------------------------------
check_var() {
    eval val=\$$1
    if [ -z "$val" ]; then
        echo "ERROR: Required environment variable '$1' is not set."
        exit 1
    fi
}

# -------------------------------------------------------
# 1. Validate required environment variables
# -------------------------------------------------------
echo "Validating environment variables..."

# PostgreSQL — datasource factory + /db-health
check_var POSTGRESQL_HOST
check_var POSTGRESQL_PORT
check_var POSTGRESQL_DB
check_var POSTGRESQL_USER
check_var POSTGRESQL_PASSWORD

# Active LLM provider — Azure AI Foundry
check_var AZURE_AI_FOUNDRY_API_KEY
check_var AZURE_AI_FOUNDRY_ENDPOINT

# Embeddings — embedding_connection.py reads directly from env
check_var EMBEDDING_API_KEY
check_var EMBEDDING_ENDPOINT
check_var EMBEDDING_MODEL_NAME
check_var EMBEDDING_API_VERSION

# Prompt files — router.py and prompt_loader.py
check_var ROUTER_PROMPT_PATH
check_var SYSTEM_PROMPT_PATH

echo "Environment validation passed."

# -------------------------------------------------------
# 2. Wait for PostgreSQL to accept TCP connections
#    server.py catches datasource init failures silently
#    (lifespan except block) — the app would start but be
#    broken. This loop ensures the DB is reachable first.
#    30 attempts x 2s = 60s max wait.
# -------------------------------------------------------
echo "Waiting for PostgreSQL at $POSTGRESQL_HOST:$POSTGRESQL_PORT..."

python -c "
import socket, time, sys
host = '$POSTGRESQL_HOST'
port = int('$POSTGRESQL_PORT')
for i in range(30):
    try:
        sock = socket.create_connection((host, port), timeout=2)
        sock.close()
        print('PostgreSQL is ready.')
        sys.exit(0)
    except OSError:
        print(f'Attempt {i+1}/30 — not ready, retrying in 2s...')
        time.sleep(2)
print('ERROR: PostgreSQL did not become ready after 60s.')
sys.exit(1)
"

# -------------------------------------------------------
# 3. Hand off to CMD
#    exec replaces this shell process so SIGTERM from
#    docker stop reaches uvicorn directly for a clean
#    graceful shutdown.
# -------------------------------------------------------
echo "Starting application..."
exec "$@"