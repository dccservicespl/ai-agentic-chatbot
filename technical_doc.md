# 🤖 AI Agentic Chatbot — Technical & Process Flow Documentation

> **Version:** 1.0.0 | **Stack:** FastAPI · LangGraph · LangChain · Azure OpenAI · PostgreSQL/pgvector · MySQL · SQLAlchemy

---

## 📋 Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Repository Structure](#3-repository-structure)
4. [Configuration & Environment](#4-configuration--environment)
5. [Application Startup Sequence](#5-application-startup-sequence)
6. [Running the Application](#6-running-the-application)
7. [API Reference](#7-api-reference)
8. [Agent Workflow — Main Graph](#8-agent-workflow--main-graph)
9. [SQL Subgraph — Deep Dive](#9-sql-subgraph--deep-dive)
10. [Infrastructure Layer](#10-infrastructure-layer)
11. [Schema Intelligence](#11-schema-intelligence)
12. [Visualization Engine](#12-visualization-engine)
13. [Logging System](#13-logging-system)
14. [Data Models Reference](#14-data-models-reference)
15. [End-to-End Request Flow](#15-end-to-end-request-flow)
16. [Dependencies](#16-dependencies)
17. [Testing](#17-testing)
18. [Change Log](#18-change-log)

---

## 1. 🌐 System Overview

The **AI Agentic Chatbot** is a production-grade, multi-agent conversational AI system that transforms natural language questions into SQL queries, executes them against a configured database, and returns structured, visualizable results — all streamed in real time.

### 🎯 Core Capabilities

| Capability | Description |
|---|---|
| 🧠 **Intent Classification** | Routes user messages: greeting · SQL query · nonsense · ambiguous |
| 🔍 **Semantic Schema Discovery** | Finds relevant DB tables via vector similarity search |
| ✍️ **LLM SQL Generation** | Azure OpenAI GPT-4o generates safe, accurate SQL |
| ✅ **Query Validation** | Syntax checks + SQL injection prevention before execution |
| ⚙️ **Multi-DB Execution** | Supports MySQL, PostgreSQL, Azure SQL, SQLite, AWS RDS |
| 📊 **Smart Visualization** | Heuristic chart type selection (bar, line, pie, KPI, table) |
| 📡 **Real-time Streaming** | Server-Sent Events (SSE) for progressive response delivery |
| 🔁 **Self-Healing Retry** | Auto-retries SQL generation on execution failure |
| 💬 **Multi-turn Memory** | Conversation persistence via LangGraph MemorySaver |

---

## 2. 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT (Browser / API)                      │
│                    POST /stream  ──  SSE Response                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI Server  (server.py)                      │
│   /health  /db-health  /schemaJson  /schemaText  /ingest  /stream   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              LangGraph Main Graph  (agent/graph.py)                 │
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌─────────────────────┐  │
│   │ router_node  │───▶│ greeting     │───▶│        END          │  │
│   │              │    └──────────────┘    └─────────────────────┘  │
│   │  (RouterNode)│    ┌──────────────┐    ┌─────────────────────┐  │
│   │              │───▶│ fallback     │───▶│        END          │  │
│   │  · FAST LLM  │    └──────────────┘    └─────────────────────┘  │
│   │  · Schema    │    ┌──────────────┐    ┌─────────────────────┐  │
│   │    Summary   │───▶│clarification │───▶│        END          │  │
│   └──────┬───────┘    └──────────────┘    └─────────────────────┘  │
│          │                                                          │
│          │ sql_query                                                │
│          ▼                                                          │
│   ┌──────────────────────────────────────────────────────────────┐ │
│   │              sql_query_node  (SQL Subgraph)                  │ │
│   │                                                              │ │
│   │  retrieve_schemas ──▶ generate_sql ──▶ validate ──▶ execute  │ │
│   │                           ▲                          │       │ │
│   │                           └──── retry on error ──────┘       │ │
│   └──────────────────────────────────────────────────────────────┘ │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────┐                                                  │
│   │  visualizer  │ ← pandas heuristics → chart type selection      │
│   └──────┬───────┘                                                  │
└──────────┼──────────────────────────────────────────────────────────┘
           │ JSON SSE
           ▼
      { content, visualization }

┌───────────────────────────┐    ┌──────────────────────────────────┐
│   Azure OpenAI (LLM)      │    │   PostgreSQL + pgvector           │
│   · FAST: gpt-4o-mini     │    │   · Schema embeddings             │
│   · SMART: gpt-4o         │    │   · Conversation memory           │
│   · Embedding: ada-002    │    └──────────────────────────────────┘
└───────────────────────────┘
┌───────────────────────────────────────────────────────────────────┐
│   Business Database (MySQL / PostgreSQL / Azure SQL / SQLite)     │
│   · Query execution target                                        │
└───────────────────────────────────────────────────────────────────┘
```

---

## 3. 📁 Repository Structure

```
ai-agentic-chatbot/
│
├── 📄 config.yaml                      # Master LLM + datasource config
├── 📄 config.example.yaml              # Config template (no secrets)
├── 📄 .env / .env.example              # Runtime environment variables
├── 📄 pyproject.toml                   # Poetry dependencies & metadata
├── 📄 poetry.lock                      # Locked dependency tree
│
├── 📂 src/ai_agentic_chatbot/
│   │
│   ├── 🚀 server.py                    # FastAPI app, lifespan, all endpoints
│   ├── 📋 logging_config.py            # Structured logging setup
│   │
│   ├── 📂 agent/                       # 🧠 Core agent workflow
│   │   ├── graph.py                    # Main LangGraph builder (build_graph)
│   │   ├── state.py                    # AgentState TypedDict
│   │   ├── router.py                   # RouterNode — intent classification
│   │   ├── schema.py                   # Pydantic request/response models
│   │   ├── registry.py                 # IntentType enum
│   │   │
│   │   ├── nodes/
│   │   │   └── visualizer.py           # VisualizationNode — chart selection
│   │   │
│   │   └── subgraphs/sql_query/        # 🗄️ SQL processing pipeline
│   │       ├── graph.py                # SQL subgraph builder
│   │       ├── state.py                # SQLSubgraphState TypedDict
│   │       ├── routes.py               # Routing functions (conditional edges)
│   │       └── nodes/
│   │           ├── retrieve_schemas.py # Semantic schema search
│   │           ├── generate_sql.py     # LLM SQL generation
│   │           ├── validate_query.py   # Safety + syntax checks
│   │           └── execute_query.py    # DB execution
│   │
│   ├── 📂 infrastructure/
│   │   ├── llm/                        # ⚡ LLM factory & config
│   │   │   ├── types.py                # LLMProvider, ModelType enums
│   │   │   ├── config.py               # AzureOpenAIConfig, etc.
│   │   │   ├── settings.py             # Settings loader (singleton)
│   │   │   ├── factory.py              # LLMFactory (singleton)
│   │   │   └── llm.py                  # Re-exports (get_llm, get_embedding)
│   │   │
│   │   ├── datasource/                 # 🗄️ Database factory & config
│   │   │   ├── datasource_types.py     # DataSourceProvider, DataSourceType enums
│   │   │   ├── datasource_config.py    # MySQLConfig, PostgreSQLConfig, etc.
│   │   │   ├── datasource_settings.py  # DataSourceSettings (singleton)
│   │   │   ├── factory.py              # DataSourceFactory (singleton)
│   │   │   ├── datasource_init.py      # initialize_datasources()
│   │   │   └── db_depency.py           # FastAPI DB session dependency
│   │   │
│   │   ├── vector_store/               # 🔍 pgvector store
│   │   │   └── pgvector_store.py       # PgVectorSchemaStore
│   │   │
│   │   └── embedding/                  # 🧬 Embedding utilities
│   │       └── embedding_connection.py # get_azure_openai_embedding()
│   │
│   ├── 📂 schema_extractor/            # 📐 DB introspection
│   │   ├── SchemaExtractor.py          # SQLAlchemy schema inspector
│   │   ├── SchemaExtractionConfig.py   # Extraction filters
│   │   ├── SchemaModels.py             # DatabaseSchema, TableSchema, etc.
│   │   ├── schema_loader.py            # SchemaLoader (singleton)
│   │   ├── SaveSchemaJson.py           # JSON serialization
│   │   ├── vector_schema_builder.py    # Chunk schema for embedding
│   │   └── table_schema_documentation.py # Human-readable schema docs
│   │
│   ├── 📂 application/                 # 🔌 Application use cases
│   │   ├── ingest_vector_schema.py     # ingest_schema() orchestrator
│   │   └── transform_schema_to_text.py # Schema → text
│   │
│   └── 📂 utils/
│       ├── utils.py                    # DB connection string helper
│       └── prompt_loader.py            # System prompt file loader
│
└── 📂 tests/
    ├── test_db_connection.py           # DB connectivity tests
    ├── test_config.py                  # Config parsing tests
    └── test_factory.py                 # Factory pattern tests
```

---

## 4. ⚙️ Configuration & Environment

### 4.1 `config.yaml` Structure

```yaml
llm:
  default: azure_openai.fast          # Default model key
  azure_openai:
    fast:                             # 🚀 Fast model (routing, cheap tasks)
      model_name: "gpt-4o-mini"
      api_key: "..."
      endpoint: "https://..."
      api_version: "2024-12-01-preview"
      temperature: 0.0
      max_tokens: 4000
    smart:                            # 🧠 Smart model (SQL generation)
      model_name: "gpt-4o"
      ...
    embedding:                        # 🧬 Embedding model (schema search)
      model_name: "text-embedding-ada-002"
      ...

datasources:
  default: mysql.primary
  mysql:
    primary:
      host: "..."
      port: 3306
      database: "ai_chatbot_db"
      ssl_ca: "..."                   # SSL certificate path
  postgresql:
    primary:
      host: "..."
      port: 5432
      database: "ai_chatbot_db"      # Also used for pgvector

logging:
  level: "INFO"
  console_level: "INFO"
  file_level: "DEBUG"
```

### 4.2 Environment Variables

| Variable | Purpose | Overrides |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI base URL | `llm.azure_openai.*.endpoint` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI credential | `llm.azure_openai.*.api_key` |
| `AZURE_OPENAI_API_VERSION` | API version string | `llm.azure_openai.*.api_version` |
| `MYSQL_HOST` / `MYSQL_PORT` | MySQL connection | `datasources.mysql.primary.*` |
| `MYSQL_DB` / `MYSQL_USER` / `MYSQL_PASSWORD` | MySQL credentials | `datasources.mysql.primary.*` |
| `POSTGRESQL_HOST` / `POSTGRESQL_PORT` | PostgreSQL connection | `datasources.postgresql.primary.*` |
| `POSTGRESQL_DB` / `POSTGRESQL_USER` / `POSTGRESQL_PASSWORD` | PostgreSQL credentials | `datasources.postgresql.primary.*` |
| `ROUTER_PROMPT_PATH` | Path to router system prompt file | — |

> 💡 All env vars are loaded by `python-dotenv` from `.env` at startup. They take **precedence** over values in `config.yaml`.

---

## 5. 🚀 Application Startup Sequence

```
1. Load .env file  (python-dotenv)
         │
         ▼
2. Setup structured logging  (logging_config.py)
   · console handler
   · app.log (debug)
   · error.log (errors only)
   · datasource.log (DB events)
         │
         ▼
3. FastAPI lifespan.startup
         │
         ├─▶ 3a. initialize_datasources()
         │       · Load DataSourceSettings from config.yaml
         │       · Apply env var overrides
         │       · Register each datasource with DataSourceFactory
         │       · Lazy-create SQLAlchemy engines (on first use)
         │
         └─▶ 3b. build_graph()  (LangGraph)
                 · Compile main StateGraph
                 · Attach MemorySaver checkpointer
                 · Compile SQL subgraph (no checkpointer)
         │
         ▼
4. Uvicorn starts listening  →  0.0.0.0:8000
         │
         ▼
5. FastAPI lifespan.shutdown  (on SIGTERM)
         · close_all_connections()  (DataSourceFactory cleanup)
```

---

## 6. ▶️ Running the Application

### Prerequisites

- Python `>=3.13`
- [Poetry](https://python-poetry.org/) package manager
- Network access to `postgres-chatbot-db.postgres.database.azure.com` (Azure PostgreSQL)

### Install Dependencies

```bash
poetry install
```

> The `.venv` folder is created automatically inside the project directory.

### Start the Server

```bash
poetry run uvicorn ai_agentic_chatbot.server:app --host 0.0.0.0 --port 8000 --reload
```

Or, if the virtualenv is already activated:

```bash
uvicorn ai_agentic_chatbot.server:app --host 0.0.0.0 --port 8000 --reload
```

### Access the Application

| URL | Description |
|---|---|
| `http://localhost:8000/docs` | Swagger UI — interactive API explorer |
| `http://localhost:8000/health` | Basic liveness check |
| `http://localhost:8000/db-health` | Database connectivity check |

### Notes

- Database credentials are configured in `config.yaml` — no `.env` file is required unless you need to override values via environment variables.
- The app connects to Azure PostgreSQL on startup. Ensure your machine has outbound access to port `5432` on the Azure host.
- Use `--reload` during development only; omit it in production.

---

## 7. 🔌 API Reference

### `GET /health`
> Basic server health probe.

**Response:** `{ "status": "ok" }`

---

### `GET /db-health`
> Verifies connectivity to all registered datasources.

**Response:**
```json
{ "status": "ok", "datasources": { "mysql.primary": true, "postgresql.primary": true } }
```

---

### `GET /schemaJson`
> Introspects the primary datasource and returns the full schema as structured JSON.

**Response:** `DatabaseSchema` object (tables, columns, primary keys, foreign keys).

---

### `GET /schemaText`
> Converts the extracted schema into a human-readable text document for debugging / inspection.

**Response:** Plain text schema documentation.

---

### `GET /ingest`
> 🔄 Triggers the schema ingestion pipeline.
> Extracts schema → builds chunks → embeds via Azure OpenAI → stores in pgvector.

**Response:** `{ "status": "ingested", "chunks": N }`

---

### `POST /stream`
> 📡 **Main endpoint.** Accepts a user message, runs the full agent workflow, and returns a Server-Sent Events (SSE) stream.

**Request Body:**
```json
{
  "thread_id": "user-session-abc123",
  "messages": [
    { "role": "user", "content": "Show me total sales by region for Q1 2024" }
  ]
}
```

**SSE Response Stream (JSON per event):**
```json
{
  "content": "Here are the total sales by region for Q1 2024:",
  "visualization": {
    "type": "bar_chart",
    "title": "Total Sales by Region — Q1 2024",
    "data": [{ "region": "North", "total_sales": 1200000 }],
    "columns": ["region", "total_sales"],
    "config": { "x_axis": "region", "y_axis": "total_sales" },
    "summary": "North region leads with $1.2M in sales.",
    "row_count": 5
  }
}
```

---

## 8. 🧠 Agent Workflow — Main Graph

**File:** `src/ai_agentic_chatbot/agent/graph.py`

### State: `AgentState`

```python
class AgentState(TypedDict):
    messages:        list[BaseMessage]   # Full conversation history
    next_step:       str                 # Router decision
    relevant_tables: list[str]           # Suggested table names
    visualization:   dict                # Chart config from visualizer
```

### Node Map

```
START
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  router_node  (RouterNode.classify)                 │
│                                                     │
│  · Loads system prompt from ROUTER_PROMPT_PATH      │
│  · Loads schema summary (SchemaLoader)              │
│  · Calls FAST LLM with structured output            │
│  · Output: RouterDecision { next_step, tables }     │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┼────────────────┬──────────────┐
          │ greeting   │ nonsense       │ clarification │ sql_query
          ▼            ▼                ▼               ▼
     greeting      fallback      clarification     sql_query
       node           node            node           (subgraph)
          │            │                │               │
          └────────────┴────────────────┴───────────────┘
                                                        │
                                                        ▼
                                                  visualizer_node
                                                        │
                                                        ▼
                                                       END
```

### Routing Logic (RouterNode)

```
1. Check if schema is loaded + tables exist
   └─ No tables in DB → next_step = "nonsense" (no data available)

2. Call FAST LLM with:
   · System prompt (loaded from file)
   · Schema summary text
   · User message

3. LLM returns RouterDecision:
   · intent: "greeting" | "sql_query" | "nonsense"
   · needs_clarification: bool
   · relevant_tables: list[str]
   · confidence: float

4. Map to next_step:
   · greeting → "greeting"
   · nonsense → "nonsense"
   · sql_query + needs_clarification → "ask_clarification"
   · sql_query → "sql_query"
```

### Checkpointing

- Uses `MemorySaver` (in-memory) checkpointer.
- Each request passes `thread_id` → enables multi-turn conversation continuity.
- State is persisted between turns within the same `thread_id`.

---

## 9. 🗄️ SQL Subgraph — Deep Dive

**File:** `src/ai_agentic_chatbot/agent/subgraphs/sql_query/graph.py`

### State: `SQLSubgraphState`

```python
class SQLSubgraphState(TypedDict):
    # ── Input ──────────────────────────────────────
    user_query:          str
    router_table_hints:  list[str]

    # ── Schema Retrieval ───────────────────────────
    retrieved_tables:    list[tuple[str, str, float]]
                         # (table_name, ddl_text, similarity_score)

    # ── SQL Generation ─────────────────────────────
    generated_sql:       str
    explanation:         str
    confidence:          float
    tables_used:         list[str]

    # ── Validation ─────────────────────────────────
    is_safe:             bool
    validation_errors:   list[str]

    # ── Execution ──────────────────────────────────
    query_result:        list[dict]
    execution_error:     str | None

    # ── Retry Control ──────────────────────────────
    generation_attempts: int
    max_retries:         int          # default: 3
```

### Pipeline Flow

```
retrieve_schemas_node
        │
        │  ✅ tables found
        ▼
generate_sql_node  ◀──────────────────────────────────────────┐
        │                                                      │
        │  ✅ no generation error                             │
        ▼                                                      │
validate_query_node                                            │
        │                                                      │
        │  ✅ is_safe = True                                  │
        ▼                                                      │
execute_query_node                                             │
        │                                                      │
        │  ❌ execution_error AND attempts < max_retries ──────┘
        │                                                      (retry with error context)
        │  ✅ success OR max_retries reached
        ▼
      END  →  returns SQLSubgraphState to parent AgentState
```

### Node Details

#### 🔍 `retrieve_schemas_node`
**File:** `nodes/retrieve_schemas.py`

1. Encodes `user_query` using Azure OpenAI embedding model.
2. Performs cosine similarity search in pgvector store.
3. Returns top-N most relevant table DDLs.
4. Expands results to include related tables via foreign key links.
5. Populates `retrieved_tables: [(name, ddl, score), ...]`.

#### ✍️ `generate_sql_node`
**File:** `nodes/generate_sql.py`

1. Uses **SMART LLM** (GPT-4o) with structured output.
2. Builds prompt with:
   - User query
   - Retrieved schema DDLs
   - (On retry) previous SQL + error message + attempt count
3. LLM returns `SQLGeneration` Pydantic model:

```python
class SQLGeneration(BaseModel):
    query:       str           # The SQL statement
    explanation: str           # Plain English reasoning
    confidence:  float         # 0.0 – 1.0
    tables_used: list[str]     # Table names referenced
    warnings:    list[str]     # Potential data concerns
```

4. Increments `generation_attempts` counter.

> ⚠️ SQL includes `LIMIT` and column count restrictions to prevent massive result sets.

#### ✅ `validate_query_node`
**File:** `nodes/validate_query.py`

Performs two categories of validation:

| Check | Description |
|---|---|
| 🛡️ **Safety** | Blocks DROP, DELETE, UPDATE, INSERT, TRUNCATE, EXEC, stored procedure calls |
| 🔤 **Syntax** | Uses `sqlparse` to validate SQL structure |

- Sets `is_safe = True` or populates `validation_errors`.

#### ⚙️ `execute_query_node`
**File:** `nodes/execute_query.py`

1. Retrieves engine from `DataSourceFactory` (configured datasource).
2. Executes validated SQL via SQLAlchemy.
3. Returns results as `list[dict]` (column: value pairs).
4. On exception, captures error into `execution_error` for retry loop.

---

## 10. 🏭 Infrastructure Layer

### 9.1 ⚡ LLM Infrastructure — Technical Deep Dive

**Package:** `src/ai_agentic_chatbot/infrastructure/llm/`

#### Package Structure

| File | Responsibility |
|---|---|
| `types.py` | `LLMProvider` and `ModelType` enums; `PROVIDER_CONFIG_REGISTRY` dict |
| `config.py` | Pydantic config models per provider; registers into `PROVIDER_CONFIG_REGISTRY` |
| `settings.py` | `Settings` / `LLMSettings` — loads `config.yaml`, applies env overrides, singleton |
| `factory.py` | `LLMFactory` — thread-safe singleton; lazy-creates and caches LangChain clients |
| `llm.py` | Module-level convenience functions: `get_llm()`, `get_embedding()` |

---

#### Types (`types.py`)

```python
class LLMProvider(Enum):
    AZURE_OPENAI = "azure_openai"
    OPENAI       = "openai"
    ANTHROPIC    = "anthropic"
    AWS_BEDROCK  = "aws_bedrock"

class ModelType(Enum):
    FAST      = "fast"       # Low-latency model — routing, cheap tasks
    SMART     = "smart"      # High-capability model — SQL generation
    EMBEDDING = "embedding"  # Embedding model — schema vectorisation
    VISION    = "vision"     # Vision-capable model (reserved)
```

`PROVIDER_CONFIG_REGISTRY` is a `Dict[LLMProvider, Type]` that maps each provider to its Pydantic config class. Entries are added by `config.py` at import time (open/closed principle — new providers register themselves).

---

#### Config Models (`config.py`)

All config classes inherit from `BaseLLMConfig` (Pydantic, `frozen=True`, `extra="forbid"`):

```
BaseLLMConfig (ABC)
    · model_name: str
    · timeout: int          (default 30s)
    · max_retries: int      (default 3)
    · get_client_kwargs() → Dict   ← abstract, returns LangChain init args
         │
         ├── AzureOpenAIConfig
         │     · api_key, endpoint (validated URL), api_version
         │     · temperature, max_tokens, top_p
         │     · frequency_penalty, presence_penalty
         │     · get_client_kwargs() → { azure_deployment, azure_endpoint, ... }
         │
         └── AzureOpenAIEmbeddingConfig  (separate — not a chat model)
               · api_key, endpoint, api_version
               · timeout, max_retries
               · get_client_kwargs() → { azure_deployment, azure_endpoint, ... }
```

`config.py` registers `AzureOpenAIConfig` into `PROVIDER_CONFIG_REGISTRY` at module load. Adding a new provider means: add an enum value in `types.py`, create a config class in `config.py`, and register it — no changes needed elsewhere.

---

#### Settings (`settings.py`) — Configuration Load & Parse Flow

```
Settings.from_config_file(config_path?)
         │
         ├─ 1. Resolve path → project_root/config.yaml
         ├─ 2. yaml.safe_load() → raw dict
         └─ 3. _parse_config(config_data)
                    │
                    ├─ Read llm.default → default_model_key
                    │   (strips provider prefix: "azure_openai.fast" → "fast")
                    │
                    └─ For each provider block in llm.*:
                           │
                           ├─ LLMProvider.from_string(provider_name)
                           ├─ get_provider_config_class(provider)  ← PROVIDER_CONFIG_REGISTRY lookup
                           │
                           └─ For each model key (fast / smart / embedding):
                                  │
                                  ├─ _apply_env_overrides(model_data, provider)
                                  │     Azure:   AZURE_OPENAI_API_KEY / ENDPOINT / API_VERSION
                                  │     OpenAI:  OPENAI_API_KEY / OPENAI_ORGANIZATION
                                  │     Anthropic: ANTHROPIC_API_KEY
                                  │     Bedrock: AWS_ACCESS_KEY_ID / SECRET / TOKEN / REGION
                                  │
                                  ├─ config_class(**model_data)  → ProviderConfig (validated)
                                  ├─ _determine_model_type(key)  → ModelType (matched by key substring)
                                  └─ ModelConfiguration(provider, model_type, config)
                                         stored in LLMSettings.models["fast" | "smart" | "embedding"]
```

The global singleton is held in a module-level `_settings` variable, exposed via `get_settings()`. Call `reload_settings()` to force a re-read from disk (e.g. after rotating API keys without restarting the process).

**`ModelConfiguration`** is the internal unit passed to the factory:

```python
class ModelConfiguration(BaseModel):
    provider:    LLMProvider    # Which provider this model belongs to
    model_type:  ModelType      # FAST / SMART / EMBEDDING / VISION
    config:      ProviderConfig # Frozen Pydantic config with all connection params
```

---

#### Factory (`factory.py`) — Client Creation & Caching Flow

`LLMFactory` is a **thread-safe double-checked locking singleton** (uses `threading.Lock`):

```
First call to LLMFactory()
    └─ __new__: acquires _lock, sets _instance
    └─ __init__: _clients = {}, _embeddings = {}, loads get_settings()

get_llm(provider?, model?)
    │
    ├─ Resolve model_key:
    │     provider=None, model=None  → settings.default_model ("fast")
    │     model only                 → model.value  ("fast" | "smart")
    │
    ├─ Cache hit?  →  return _clients[model_key]   ← no LLM call, instant
    │
    └─ Cache miss:
           settings.get_model_config(model_key) → ModelConfiguration
           _create_client(model_config)
               └─ provider == AZURE_OPENAI:
                      AzureChatOpenAI(**config.get_client_kwargs())
               └─ other providers: raises ValueError (extensible)
           _clients[model_key] = client
           return client

get_embedding(provider?, model?)
    │
    ├─ model must be ModelType.EMBEDDING (enforced)
    ├─ Cache key: "{provider}.embedding"
    ├─ Cache hit?  →  return _embeddings[key]
    └─ Cache miss:
           _get_embedding_config(provider)
               └─ settings.get_model_config("embedding") → AzureOpenAIEmbeddingConfig
           _create_embedding_client(provider, config)
               └─ AzureOpenAIEmbeddings(**config.get_client_kwargs())
           _embeddings[key] = client
           return client
```

**Cache behaviour:** LangChain client objects are expensive to construct (HTTP connection setup, SDK initialisation). The factory holds them for the process lifetime. `clear_cache()` discards all cached clients; `reload_settings()` calls `clear_cache()` then re-reads the config file.

---

#### Public API (`llm.py` / `factory.py`)

```python
from ai_agentic_chatbot.infrastructure.llm import get_llm, get_embedding
from ai_agentic_chatbot.infrastructure.llm.types import ModelType

# Router node — low-latency classification
llm = get_llm(model=ModelType.FAST)

# SQL generation node — highest-capability model
llm = get_llm(model=ModelType.SMART)

# Schema embedding / vector search
emb = get_embedding()   # defaults to AZURE_OPENAI + EMBEDDING

# Structured output (LangChain pattern used in SQL node)
structured_llm = llm.with_structured_output(SQLGeneration)
```

---

#### Provider Support Matrix

| Provider Enum | LangChain Client | Chat | Embedding | Status |
|---|---|---|---|---|
| `AZURE_OPENAI` | `AzureChatOpenAI` / `AzureOpenAIEmbeddings` | ✅ | ✅ | **Active** |
| `OPENAI` | `ChatOpenAI` | ✅ | — | Enum defined, factory raises `ValueError` (not yet wired) |
| `ANTHROPIC` | `ChatAnthropic` | ✅ | — | Enum defined, factory raises `ValueError` |
| `AWS_BEDROCK` | `BedrockChat` | ✅ | — | Enum defined, factory raises `ValueError` |

> Only `AZURE_OPENAI` is fully wired end-to-end. The other providers have enum values and env-override logic in `settings.py` but `_create_client()` in the factory will raise `ValueError` until their LangChain instantiation is added.

---

#### Environment Variable Override Reference

| Variable | Provider | Overrides |
|---|---|---|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI | `config.yaml llm.azure_openai.*.api_key` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI | `config.yaml llm.azure_openai.*.endpoint` |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI | `config.yaml llm.azure_openai.*.api_version` |
| `OPENAI_API_KEY` | OpenAI | `llm.openai.*.api_key` |
| `OPENAI_ORGANIZATION` | OpenAI | `llm.openai.*.organization` |
| `ANTHROPIC_API_KEY` | Anthropic | `llm.anthropic.*.api_key` |
| `AWS_ACCESS_KEY_ID` | AWS Bedrock | `llm.aws_bedrock.*.aws_access_key_id` |
| `AWS_SECRET_ACCESS_KEY` | AWS Bedrock | `llm.aws_bedrock.*.aws_secret_access_key` |
| `AWS_SESSION_TOKEN` | AWS Bedrock | `llm.aws_bedrock.*.aws_session_token` |
| `AWS_DEFAULT_REGION` | AWS Bedrock | `llm.aws_bedrock.*.region_name` |

> Env vars always win over `config.yaml` values. The override is applied in `Settings._apply_env_overrides()` before Pydantic validation.

---

### 9.2 🗄️ DataSource Factory

**Files:** `infrastructure/datasource/`

```
DataSourceSettings (singleton)
    · Loads config.yaml → datasources section
    · Maps provider strings to config classes
    · Applies env var overrides (MYSQL_HOST, etc.)
         │
         ▼
DataSourceFactory (singleton)
    · Registry: name → (config, engine | None)
    · Lazy-creates SQLAlchemy Engine on first get_engine()
    · Connection pooling: pool_size, max_overflow, pool_recycle
    · Supports by-type and by-provider queries
```

**Supported Database Providers:**

| Provider | Driver | Config Class |
|---|---|---|
| `MYSQL` | pymysql | `MySQLConfig` |
| `POSTGRESQL` | psycopg2 | `PostgreSQLConfig` |
| `AZURE_SQL` | ODBC | `AzureSQLConfig` |
| `AWS_RDS_MYSQL` | pymysql | `MySQLConfig` |
| `AWS_RDS_POSTGRESQL` | psycopg2 | `PostgreSQLConfig` |
| `SQLITE` | sqlite3 | `SQLiteConfig` |

**DataSource Types** (organizational tagging):

| Type | Purpose |
|---|---|
| `PRIMARY` | Main business data |
| `ANALYTICS` | Read-heavy reporting replica |
| `CACHE` | Fast-access store |
| `LOGGING` | Audit/event logs |
| `BACKUP` | Failover / DR |

---

### 9.3 🔍 Vector Store (pgvector)

**File:** `infrastructure/vector_store/pgvector_store.py`

- Class: `PgVectorSchemaStore`
- Backed by **LangChain PGVector** integration.
- Stores: table name, DDL text, column descriptions as LangChain `Document` objects.
- Similarity metric: **cosine** (default for pgvector).
- Used by `retrieve_schemas_node` for semantic table discovery.

**Ingestion Flow** (`GET /ingest`):

```
SchemaExtractor.extract_database_schema()
        │
        ▼
VectorSchemaBuilder.build_all_tables()
        │  Produces text chunks per table
        ▼
Azure OpenAI Embeddings (text-embedding-ada-002)
        │  Converts chunks to 1536-dim vectors
        ▼
PgVectorSchemaStore.ingest()
        │  Upserts vectors + metadata to PostgreSQL
        ▼
pgvector table in ai_chatbot_db
```

---

## 11. 📐 Schema Intelligence

The `schema_extractor` package is the foundation of the system's ability to understand and query a database. It handles three concerns: **live DB introspection**, **LLM-driven semantic enrichment**, and **runtime schema access** for the SQL agent.

### 10.1 Package Modules

| Module | Role |
|---|---|
| `SchemaModels.py` | Pure dataclasses — in-memory schema representation |
| `SchemaExtractionConfig.py` | Filter config (include/exclude schemas and tables) |
| `SchemaExtractor.py` | Live DB introspection via SQLAlchemy inspector |
| `SaveSchemaJson.py` | Atomic JSON serialization to `temp/db_schema.json` |
| `table_schema_documentation.py` | Pydantic model for LLM-generated table documentation |
| `vector_schema_builder.py` | Converts YAML docs to semantic text blocks for embedding |
| `schema_loader.py` | Runtime singleton — loads schema context for agent queries |

---

### 10.2 Data Models

```
DatabaseSchema
  └── tables: list[TableSchema]
        ├── schema_name: str
        ├── table_name: str
        ├── columns: list[ColumnSchema]
        │     ├── name: str
        │     ├── data_type: str
        │     ├── nullable: bool
        │     └── default: Optional[str]
        ├── primary_keys: list[str]
        └── foreign_keys: list[ForeignKeySchema]
              ├── column: str
              ├── referred_table: str
              └── referred_column: str
```

---

### 10.3 Schema Extraction (`SchemaExtractor`)

**File:** `schema_extractor/SchemaExtractor.py`

Uses SQLAlchemy `Inspector` to reflect a live database, filtered by `SchemaExtractionConfig`:

```
SchemaExtractor(engine, config)
  └── extract_database_schema()
        ├── _get_schemas()           → list[str]  (falls back to ["public"])
        ├── _schema_allowed()        → applies include_schemas whitelist
        ├── _table_allowed()         → applies include_tables / exclude_tables filters
        └── Per allowed table:
              ├── _extract_columns()       → list[ColumnSchema]
              ├── _extract_primary_keys()  → list[str]
              └── _extract_foreign_keys()  → list[ForeignKeySchema]
        └── Returns DatabaseSchema
```

**`SchemaExtractionConfig` options:**

| Field | Type | Effect |
|---|---|---|
| `include_schemas` | `list[str] \| None` | Whitelist DB schemas; if `None`, all schemas are included |
| `include_tables` | `list[str] \| None` | Whitelist table names |
| `exclude_tables` | `list[str] \| None` | Blacklist table names |

---

### 10.4 Schema Persistence (`SaveSchemaJson`)

**File:** `schema_extractor/SaveSchemaJson.py`

- `save_schema_temp_file(schema)` — serializes `DatabaseSchema` to dict via `dataclasses.asdict()` and atomically writes to `temp/db_schema.json` using `os.replace()` (crash-safe on all major OS).
- `write_text_file()` / `serialize_data()` — generic utilities supporting `json`, `yaml`, and `text` output formats used by downstream steps.

---

### 10.5 LLM-Generated Documentation (`TableSchemaDocumentation`)

**File:** `schema_extractor/table_schema_documentation.py`

The raw schema JSON is fed to an LLM, which produces a `TableSchemaDocumentation` Pydantic model per table. This enriches structural data with business-level semantics:

```python
class TableSchemaDocumentation(BaseModel):
    table_name:          str
    business_purpose:    str               # "why does this table exist"
    primary_identifier:  str               # PK explanation in plain English
    key_fields:          list[KeyField]    # field_name + business meaning
    important_dates:     list[ImportantDate] | None
    relationships:       list[RelationshipExplanation] | None  # FK → plain English
    operational_notes:   str | None        # flags, statuses, enums
    example_questions:   list[str]         # natural language Q&A hints
```

> Uses `model_config = {"extra": "forbid"}` to enforce strict structured output from the LLM.

---

### 10.6 Vector Text Builder (`VectorSchemaBuilder`)

**File:** `schema_extractor/vector_schema_builder.py`

Converts the LLM-generated YAML documentation into rich text blocks for embedding:

```
VectorSchemaBuilder.build_all_tables(schema_yaml)
  └── Per table → build_table_text()
        Assembles sections:
          · Database / Schema Version
          · Table Name + Business Purpose
          · Primary Identifier
          · Key Business Fields
          · Important Dates
          · Relationships
          · Operational Notes
          · Typical Questions This Table Answers
        └── Returns list of:
              { table_name, content (text block), metadata }
```

The `metadata` dict (`database`, `schema_version`, `table_name`, `object_type`) is stored alongside each vector in pgvector for filtering.

---

### 10.7 Runtime Schema Loader (`SchemaLoader`)

**File:** `schema_extractor/schema_loader.py`

Singleton accessed via `get_schema_loader()`. Used by the SQL agent at query time:

| Method | What it loads | Source |
|---|---|---|
| `load_schema_json()` | Raw structural schema | `temp/db_schema.json` |
| `load_schema_documentation()` | LLM-enriched YAML docs | `$SCHEMA_PATH` env var |
| `load_schema_summary()` | Compact summary for router hints | `$SCHEMA_SUMMARY_PATH` env var |
| `get_table_docs_for_search()` | Rich search documents for vector retrieval | YAML docs (with fallback to raw JSON) |

`get_table_docs_for_search()` builds per-table documents with:
- `search_text` — concatenated purpose, fields, relationships, and example questions
- `ddl` — reconstructed `CREATE TABLE` statement (types inferred via `_infer_data_type()` heuristics)
- `columns`, `key_fields`, `relationships`, `example_questions`

> **Fallback:** if `$SCHEMA_PATH` YAML is missing, it degrades gracefully to raw JSON with reduced semantic context.

> **Known limitation (TODO in code):** `get_schema_loader()` returns a module-level singleton — not safe for multi-tenant deployments where different tenants have different schemas.

---

### 10.8 The `/schemaJson` API Endpoint and Its Role

**File:** `server.py` — `GET /schemaJson`

This endpoint is the **trigger for Step 1** of a one-time schema setup pipeline. It wires together `SchemaExtractionConfig`, `SchemaExtractor`, and `SaveSchemaJson`:

```python
config = SchemaExtractionConfig(
    include_tables=["orders", "customer", "sales", "product", "inventory"]
)
extractor = SchemaExtractor(db_engine, config)   # db_engine injected via Depends(get_engine)
schema    = extractor.extract_database_schema()
path      = save_schema_temp_file(schema)
# → Response: { "SchemaPath": "...temp/db_schema.json" }
```

---

### 10.9 Full Schema Setup Pipeline (3-Step)

These three endpoints must be called **once during initial setup** to build the vector-searchable schema context:

```
Step 1 — GET /schemaJson
    SchemaExtractionConfig (table whitelist)
         │
    SchemaExtractor → live DB introspection (SQLAlchemy)
         │
    save_schema_temp_file()
         └─► temp/db_schema.json

Step 2 — GET /schemaText
    transform_schema_to_text()
         · Reads temp/db_schema.json
         · Sends each table to LLM → TableSchemaDocumentation
         └─► schema_documentation.yaml  ($SCHEMA_PATH)

Step 3 — GET /ingest
    ingest_schema(schema_path, pg_conn_str)
         · VectorSchemaBuilder.build_all_tables()  → text chunks
         · Azure OpenAI text-embedding-ada-002      → 1536-dim vectors
         · PgVectorSchemaStore.ingest()             → pgvector (PostgreSQL)
         └─► Schema is now semantically searchable by the SQL agent
```

After setup, the SQL agent uses `SchemaLoader.get_table_docs_for_search()` at query time to retrieve the most relevant tables via cosine similarity search.

---

## 12. 📊 Visualization Engine

**File:** `agent/nodes/visualizer.py` — `VisualizationNode`

### Chart Type Heuristics

The visualizer analyzes the `query_result` DataFrame and applies these rules in order:

| Condition | Chart Type |
|---|---|
| 1 row, 1 column | 🔢 **KPI** (single metric display) |
| 2 columns: date/time + numeric | 📈 **Line Chart** |
| 2 columns: categorical + numeric | 📊 **Bar Chart** |
| 2 columns: column name contains "percent" | 🥧 **Pie Chart** |
| 3+ columns OR multi-row | 📋 **Table** |
| Fallback / text response | 💬 **Text** |

### Value Formatting

| Column Name Pattern | Format Applied |
|---|---|
| `sales`, `revenue`, `amount`, `price`, `cost` | 💰 Currency (`$1,234,567`) |
| `percent`, `rate`, `ratio` + value in 0–1 | 📊 Percentage (`12.5%`) |
| `count`, `number`, `qty`, `total` | 🔢 Integer (`1,234`) |
| Other numeric | Decimal (`.2f`) |

### Visualization Payload Schema

```json
{
  "type": "bar_chart | line_chart | pie_chart | kpi | table | text",
  "title": "Human-readable chart title",
  "data": [
    { "column_a": "value", "column_b": 42 }
  ],
  "columns": ["column_a", "column_b"],
  "config": {
    "x_axis": "column_a",
    "y_axis": "column_b",
    "label_column": "column_a",
    "value_column": "column_b"
  },
  "summary": "One-sentence insight about the data",
  "row_count": 10
}
```

---

## 13. 📋 Logging System

**File:** `logging_config.py`

### Handlers

| Handler | File | Level | Format |
|---|---|---|---|
| `console` | stdout | INFO | Human-readable |
| `file` | `app.log` | DEBUG | JSON structured |
| `error_file` | `error.log` | ERROR | JSON structured |
| `datasource_file` | `datasource.log` | DEBUG | JSON structured |

### Usage in Modules

```python
from logging_config import get_logger

logger = get_logger(__name__)
logger.info("SQL generated", extra={"sql": generated_sql, "confidence": 0.95})
```

### Log Levels by Component

| Component | Recommended Level |
|---|---|
| Router decisions | INFO |
| SQL generation | DEBUG |
| SQL validation failures | WARNING |
| Execution errors | ERROR |
| DB connection events | datasource handler |

---

## 14. 📦 Data Models Reference

### Request / Response Models (`agent/schema.py`)

```python
class Message(BaseModel):
    role:    str    # "user" | "assistant" | "system"
    content: str

class StreamRequest(BaseModel):
    thread_id: str           # Conversation session identifier
    messages:  list[Message] # Full message history

class IntentResult(BaseModel):
    intent:     str    # "greeting" | "sql_query" | "nonsense"
    confidence: float  # 0.0 – 1.0
```

### Router Decision (structured LLM output)

```python
class RouterDecision(BaseModel):
    intent:               str        # Classification result
    needs_clarification:  bool       # True → ask user for more detail
    relevant_tables:      list[str]  # Predicted table names
    confidence:           float      # Model confidence
```

### SQL Generation (structured LLM output)

```python
class SQLGeneration(BaseModel):
    query:       str         # SELECT statement
    explanation: str         # Plain English reasoning
    confidence:  float       # 0.0 – 1.0
    tables_used: list[str]   # Table names in query
    warnings:    list[str]   # Data caveats
```

---

## 15. 🔄 End-to-End Request Flow

### Example: *"Show me total sales by region for Q1 2024"*

```
① Client POSTs to /stream
   { thread_id: "abc", messages: [{ role: "user", content: "..." }] }

② FastAPI invokes graph.astream(state, config={ thread_id })

③ router_node
   · Loads schema summary → confirms sales/region tables exist
   · Calls gpt-4o-mini with RouterDecision schema
   · Returns: intent="sql_query", relevant_tables=["orders","regions"]
   · next_step = "sql_query"

④ sql_query_node (subgraph)

   ④a retrieve_schemas
       · Embeds "total sales by region Q1 2024"
       · Cosine search in pgvector → returns orders, regions DDLs
       · Score: orders=0.92, regions=0.87

   ④b generate_sql (attempt 1)
       · Calls gpt-4o with schema context
       · Returns:
         query: "SELECT r.region_name, SUM(o.amount) AS total_sales
                 FROM orders o JOIN regions r ON o.region_id = r.id
                 WHERE o.order_date BETWEEN '2024-01-01' AND '2024-03-31'
                 GROUP BY r.region_name
                 ORDER BY total_sales DESC LIMIT 100"
         confidence: 0.94

   ④c validate_query
       · No DROP/DELETE/UPDATE found ✅
       · sqlparse validates syntax ✅
       · is_safe = True

   ④d execute_query
       · DataSourceFactory.get_engine("mysql.primary")
       · Executes SQL → 5 rows returned
       · query_result = [{ "region_name": "North", "total_sales": 1200000 }, ...]

⑤ visualizer_node
   · DataFrame: 2 columns (categorical + numeric)
   · Heuristic → bar_chart
   · Format total_sales → currency ($1,200,000)
   · Generates title: "Total Sales by Region — Q1 2024"
   · Builds visualization payload

⑥ graph yields final state via astream

⑦ FastAPI SSE sends JSON event to client:
   {
     "content": "Here are the total sales by region for Q1 2024:",
     "visualization": {
       "type": "bar_chart",
       "title": "Total Sales by Region — Q1 2024",
       "data": [...],
       "config": { "x_axis": "region_name", "y_axis": "total_sales" },
       "summary": "North region leads with $1.2M.",
       "row_count": 5
     }
   }
```

### Retry Scenario

```
④d execute_query → ❌ "Unknown column 'region_id' in 'on clause'"
   execution_error set, generation_attempts = 1 (< max_retries=3)
        │
        ▼
④b generate_sql (attempt 2)
   · Prompt now includes: previous_sql + "Error: Unknown column 'region_id'..."
   · LLM corrects join condition
   · Returns new SQL with correct column name
        │
        ▼
④c validate → ✅
④d execute → ✅ success
```

---

## 16. 📚 Dependencies

### Core Runtime

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.128.0 | HTTP framework |
| `uvicorn` | 0.40.0 | ASGI server |
| `langchain` | 1.2.6 | LLM abstraction layer |
| `langchain-community` | 0.4.1 | Community integrations |
| `langchain-openai` | 1.1.7 | Azure OpenAI / OpenAI clients |
| `langgraph` | 1.0.7 | Graph-based workflow orchestration |
| `langchain-postgres` | 0.0.16 | pgvector integration |
| `sqlalchemy` | 2.0.46 | Database ORM / engine |
| `pymysql` | 1.1.2 | MySQL sync driver |
| `aiomysql` | 0.3.2 | MySQL async driver |
| `psycopg2-binary` | 2.9.11 | PostgreSQL driver |
| `pydantic` | 2.0.0 | Data validation / structured output |
| `pyyaml` | 6.0.0 | config.yaml parsing |
| `python-dotenv` | 1.0.0 | `.env` loading |
| `pandas` | 2.0.0 | DataFrame analysis for visualization |
| `sqlparse` | 0.4.0 | SQL syntax validation |
| `cryptography` | 46.0.4 | SSL encryption |
| `python-json-logger` | 4.0.0 | Structured JSON logging |

### Development

| Package | Version | Purpose |
|---|---|---|
| `pytest` | 9.0.2 | Test runner |
| `pytest-asyncio` | 1.3.0 | Async test support |

> **Python requirement:** `>=3.13`  
> **Package manager:** Poetry

---

## 17. 🧪 Testing

### Test Files

| File | Coverage |
|---|---|
| `tests/test_db_connection.py` | Database connectivity for all registered datasources |
| `tests/test_config.py` | Config YAML parsing, env var overrides, defaults |
| `tests/test_factory.py` | LLMFactory and DataSourceFactory singleton behavior |

### Running Tests

```bash
# Run all tests
poetry run pytest

# Verbose output
poetry run pytest -v

# Specific test file
poetry run pytest tests/test_db_connection.py -v

# With coverage
poetry run pytest --cov=src/ai_agentic_chatbot
```

### Key Test Scenarios

- ✅ All datasources connect and respond within timeout
- ✅ Config merges correctly with env var overrides
- ✅ LLMFactory returns cached instance on repeated calls
- ✅ DataSourceFactory reuses engines (no double-init)
- ✅ SQL validation blocks injection patterns

---

## 🔐 Security Notes

| Risk | Mitigation |
|---|---|
| SQL Injection | `validate_query_node` blocks DML/DDL keywords before execution |
| Credential Exposure | Secrets in `.env` (gitignored), never in config.yaml or source |
| Unbounded Queries | SQL generation enforced with `LIMIT` and column count caps |
| Prompt Injection | System prompts loaded from files, not constructed from user input |
| DB Privileges | Runtime DB user should have `SELECT` only on business tables |

---

## 📝 Glossary

| Term | Definition |
|---|---|
| **LangGraph** | Graph-based workflow framework where nodes are Python functions and edges are conditional routing decisions |
| **pgvector** | PostgreSQL extension for storing and querying high-dimensional embedding vectors |
| **MemorySaver** | LangGraph in-memory checkpointer that persists graph state per `thread_id` |
| **Structured Output** | LangChain feature (`.with_structured_output()`) that forces LLM responses to conform to a Pydantic model |
| **SSE** | Server-Sent Events — one-way HTTP streaming from server to client |
| **DDL** | Data Definition Language — the `CREATE TABLE` statements stored alongside schema embeddings |
| **Singleton** | Design pattern ensuring only one instance of a class (Factory, Settings) exists per process |
| **Cosine Similarity** | Distance metric used by pgvector to rank semantic similarity between query embeddings and schema embeddings |

---

---

## 18. 📋 Change Log

### v1.1.0 — Azure AI Foundry Model Migration (2026-06-05)

**Context:** The LLM backend is switching from Azure OpenAI Service (GPT-4o-mini / GPT-4o) to Azure AI Foundry serverless deployments (DeepSeek-V4-Flash / Llama-3.3-70B-Instruct). These models are hosted on a different endpoint type (`*.services.ai.azure.com/openai/v1/`) that requires `ChatOpenAI` (OpenAI-compatible client) rather than `AzureChatOpenAI`, and does not use `api_version`.

#### Root Cause

| Issue | Detail |
|---|---|
| Wrong LangChain client | `AzureChatOpenAI` is for Azure OpenAI Service; Azure AI Foundry serverless needs `ChatOpenAI` with `base_url` |
| `azure_deployment` vs `model` | `AzureChatOpenAI` uses `azure_deployment`; `ChatOpenAI` uses `model` parameter |
| Empty `api_version` | Azure AI Foundry endpoints have no API version; current config has `api_version: ""` which is invalid for `AzureChatOpenAI` |
| `strict=True` incompatibility | `strict=True` in `.with_structured_output()` is an OpenAI-only JSON schema enforcement flag; DeepSeek and Llama do not support it and will return API errors |

#### Pending Implementation Steps

| # | File | Change Required |
|---|---|---|
| 1 | `infrastructure/llm/types.py` | Add `AZURE_AI_FOUNDRY = "azure_ai_foundry"` to `LLMProvider` enum |
| 2 | `infrastructure/llm/config.py` | Add `AzureAIFoundryConfig` class whose `get_client_kwargs()` returns `ChatOpenAI`-compatible kwargs (`base_url`, `model`, `api_key`); register in `PROVIDER_CONFIG_REGISTRY`; update `ProviderConfig` Union |
| 3 | `infrastructure/llm/factory.py` | Add `_create_azure_ai_foundry_client()` using `ChatOpenAI(**config.get_client_kwargs())`; add branch in `_create_client()` for `LLMProvider.AZURE_AI_FOUNDRY` |
| 4 | `infrastructure/llm/settings.py` | Add `elif provider == LLMProvider.AZURE_AI_FOUNDRY:` block in `_apply_env_overrides()` mapping `AZURE_AI_FOUNDRY_API_KEY` and `AZURE_AI_FOUNDRY_ENDPOINT` |
| 5 | `config.yaml` | Move `fast:` and `smart:` model entries from `azure_openai:` to a new `azure_ai_foundry:` key; update `llm.default` to `azure_ai_foundry.fast`; remove `api_version: ""`; keep `embedding:` under `azure_openai:` (still uses real Azure OpenAI endpoint) |
| 6 | `application/transform_schema_to_text.py:38`<br>`agent/router.py:69`<br>`agent/subgraphs/sql_query/nodes/generate_sql.py:48` | Remove `strict=True` from all three `.with_structured_output()` calls |

#### Model Change Summary

| Role | Previous Model | New Model | Provider Type |
|---|---|---|---|
| Fast (routing) | `gpt-4o-mini` via Azure OpenAI | `DeepSeek-V4-Flash` via Azure AI Foundry | `AZURE_AI_FOUNDRY` |
| Smart (SQL gen) | `gpt-4o` via Azure OpenAI | `Llama-3.3-70B-Instruct` via Azure AI Foundry | `AZURE_AI_FOUNDRY` |
| Embedding | `text-embedding-3-small` via Azure OpenAI | No change | `AZURE_OPENAI` |

#### Architecture Diagram Update (after migration)

```
┌───────────────────────────────────────────────────────────────┐
│   Azure AI Foundry Serverless (LLM)                           │
│   Endpoint: dccglobal-ai-services.services.ai.azure.com       │
│   · FAST:  DeepSeek-V4-Flash    (routing, cheap tasks)        │
│   · SMART: Llama-3.3-70B-Instruct (SQL generation)            │
└───────────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────────┐
│   Azure OpenAI Service (Embedding — unchanged)                │
│   · Embedding: text-embedding-3-small                         │
└───────────────────────────────────────────────────────────────┘
```

---

*Generated: 2026-05-29 | Updated: 2026-06-05 | Repository: `ai-agentic-chatbot` | Branch: `develop`*
