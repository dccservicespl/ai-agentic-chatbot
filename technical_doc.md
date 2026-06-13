# 🤖 AI Agentic Chatbot — Technical & Process Flow Documentation

> **Version:** 1.1.0 | **Stack:** FastAPI · LangGraph · LangChain · Azure AI Foundry · Azure OpenAI · PostgreSQL/pgvector · SQLAlchemy

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
19. [Known Performance Issue — Excessive Embedding API Calls](#19-known-performance-issue--excessive-embedding-api-calls)
20. [Dockerize & Deploy to Azure VM — Todo List](#20-dockerize--deploy-to-azure-vm--todo-list)
21. [NL-to-SQL Accuracy Improvements — Improvement Plan](#21-nl-to-sql-accuracy-improvements--improvement-plan)

---

## 1. 🌐 System Overview

The **AI Agentic Chatbot** is a production-grade, multi-agent conversational AI system that transforms natural language questions into SQL queries, executes them against a configured database, and returns structured, visualizable results — all streamed in real time.

### 🎯 Core Capabilities

| Capability | Description |
|---|---|
| 🧠 **Intent Classification** | Routes user messages: greeting · SQL query · nonsense · ambiguous |
| 🔍 **Semantic Schema Discovery** | Finds relevant DB tables via vector similarity search |
| ✍️ **LLM SQL Generation** | Llama-3.3-70B-Instruct (Azure AI Foundry) generates safe, accurate SQL |
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

┌──────────────────────────────────────┐    ┌──────────────────────────────────┐
│   Azure AI Foundry Serverless (LLM)  │    │   PostgreSQL + pgvector           │
│   · FAST:  DeepSeek-V4-Flash         │    │   · Schema embeddings             │
│   · SMART: Llama-3.3-70B-Instruct    │    │   · Conversation memory           │
└──────────────────────────────────────┘    └──────────────────────────────────┘
┌──────────────────────────────────────┐
│   Azure OpenAI Service (Embedding)   │
│   · text-embedding-3-small           │
└──────────────────────────────────────┘
┌───────────────────────────────────────────────────────────────────┐
│   Business Database (PostgreSQL / Azure SQL / SQLite)             │
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
  default: azure_ai_foundry.fast      # Active default — DeepSeek-V4-Flash

  azure_ai_foundry:                   # Azure AI Foundry serverless (ChatOpenAI client)
    fast:                             # 🚀 Fast model — routing, intent classification
      model_name: "DeepSeek-V4-Flash"
      api_key: "..."
      endpoint: "https://dccglobal-ai-services.services.ai.azure.com/openai/v1/"
      temperature: 0.1
      max_retries: 3
    smart:                            # 🧠 Smart model — SQL generation
      model_name: "Llama-3.3-70B-Instruct"
      api_key: "..."
      endpoint: "https://dccglobal-ai-services.services.ai.azure.com/openai/v1/"
      temperature: 0.1
      max_tokens: 20000
      timeout: 120
      max_retries: 3

  azure_openai:                       # Azure OpenAI Service (AzureChatOpenAI client)
    fast:                             # Fallback — gpt-4o-mini (inactive, overridden by Foundry)
      model_name: "gpt-4o-mini"
      api_key: "..."
      endpoint: "https://dcc-azure-openai.cognitiveservices.azure.com"
      api_version: "2024-12-01-preview"
      temperature: 0.1
    smart:                            # Fallback — gpt-4.1 (inactive, overridden by Foundry)
      model_name: "gpt-4.1"
      api_key: "..."
      endpoint: "https://dcc-azure-openai.cognitiveservices.azure.com"
      api_version: "2024-12-01-preview"
      temperature: 0.1
      max_tokens: 32768
    embedding:                        # 🧬 Embedding model — schema search (active)
      model_name: "text-embedding-3-small"
      api_key: "..."
      endpoint: "https://dcc-azure-openai.cognitiveservices.azure.com"
      api_version: "2023-05-15"

datasources:
  default: postgresql.primary
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

> **Provider resolution note:** `settings.py` stores models by short key (`fast`, `smart`, `embedding`). When both `azure_openai` and `azure_ai_foundry` declare `fast`/`smart`, the last-parsed provider wins. Since `azure_ai_foundry` is listed first in `config.yaml`, it takes precedence. To switch back to Azure OpenAI models, move the `azure_openai` block above `azure_ai_foundry`.

### 4.2 Environment Variables

| Variable | Purpose | Overrides |
|---|---|---|
| `AZURE_AI_FOUNDRY_API_KEY` | Azure AI Foundry credential (fast + smart) | `llm.azure_ai_foundry.*.api_key` |
| `AZURE_AI_FOUNDRY_ENDPOINT` | Azure AI Foundry base URL | `llm.azure_ai_foundry.*.endpoint` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI base URL (embedding + fallback) | `llm.azure_openai.*.endpoint` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI credential (embedding + fallback) | `llm.azure_openai.*.api_key` |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI API version string | `llm.azure_openai.*.api_version` |
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
    AZURE_OPENAI     = "azure_openai"      # Azure OpenAI Service (AzureChatOpenAI)
    AZURE_AI_FOUNDRY = "azure_ai_foundry"  # Azure AI Foundry serverless (ChatOpenAI)
    OPENAI           = "openai"
    ANTHROPIC        = "anthropic"
    AWS_BEDROCK      = "aws_bedrock"

class ModelType(Enum):
    FAST      = "fast"       # Low-latency model — routing, intent classification
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
         ├── AzureOpenAIConfig                       ← AZURE_OPENAI provider
         │     · api_key, endpoint (validated URL), api_version
         │     · temperature, max_tokens, top_p
         │     · frequency_penalty, presence_penalty
         │     · get_client_kwargs() → { azure_deployment, azure_endpoint, api_version, ... }
         │
         ├── AzureAIFoundryConfig                    ← AZURE_AI_FOUNDRY provider
         │     · api_key, endpoint (validated URL)
         │     · temperature, max_tokens, top_p
         │     · frequency_penalty, presence_penalty
         │     · get_client_kwargs() → { model, base_url, api_key, ... }
         │       (no api_version — Foundry endpoints are version-free)
         │
         └── AzureOpenAIEmbeddingConfig  (separate — not a chat model)
               · api_key, endpoint, api_version
               · timeout, max_retries
               · get_client_kwargs() → { azure_deployment, azure_endpoint, ... }
```

`config.py` registers both `AzureOpenAIConfig` and `AzureAIFoundryConfig` into `PROVIDER_CONFIG_REGISTRY` at module load. Adding a new provider means: add an enum value in `types.py`, create a config class in `config.py`, and register it — no changes needed elsewhere.

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
                                  │     Foundry:  AZURE_AI_FOUNDRY_API_KEY / ENDPOINT
                                  │     Azure:    AZURE_OPENAI_API_KEY / ENDPOINT / API_VERSION
                                  │     OpenAI:   OPENAI_API_KEY / OPENAI_ORGANIZATION
                                  │     Anthropic: ANTHROPIC_API_KEY
                                  │     Bedrock:  AWS_ACCESS_KEY_ID / SECRET / TOKEN / REGION
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
               └─ provider == AZURE_AI_FOUNDRY:
                      ChatOpenAI(**config.get_client_kwargs())  ← base_url + model
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
| `AZURE_AI_FOUNDRY` | `ChatOpenAI` (base_url) | ✅ | — | **Active — default provider** |
| `AZURE_OPENAI` | `AzureChatOpenAI` / `AzureOpenAIEmbeddings` | ✅ | ✅ | **Active — embedding only (chat overridden by Foundry)** |
| `OPENAI` | `ChatOpenAI` | ✅ | — | Enum defined, factory raises `ValueError` (not yet wired) |
| `ANTHROPIC` | `ChatAnthropic` | ✅ | — | Enum defined, factory raises `ValueError` |
| `AWS_BEDROCK` | `BedrockChat` | ✅ | — | Enum defined, factory raises `ValueError` |

> `AZURE_AI_FOUNDRY` is the active chat provider (DeepSeek + Llama). `AZURE_OPENAI` remains active for embedding only. To switch back to Azure OpenAI chat models, reorder the provider blocks in `config.yaml` so `azure_openai` is parsed last.

---

#### Environment Variable Override Reference

| Variable | Provider | Overrides |
|---|---|---|
| `AZURE_AI_FOUNDRY_API_KEY` | Azure AI Foundry | `config.yaml llm.azure_ai_foundry.*.api_key` |
| `AZURE_AI_FOUNDRY_ENDPOINT` | Azure AI Foundry | `config.yaml llm.azure_ai_foundry.*.endpoint` |
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

### v1.1.0 — Azure AI Foundry Model Migration (2026-06-05) ✅ COMPLETED

**Context:** The LLM backend was migrated from Azure OpenAI Service (GPT-4o-mini / GPT-4.1) to Azure AI Foundry serverless deployments (DeepSeek-V4-Flash / Llama-3.3-70B-Instruct). These models are hosted on a different endpoint type (`*.services.ai.azure.com/openai/v1/`) that requires `ChatOpenAI` (OpenAI-compatible client) rather than `AzureChatOpenAI`, and does not use `api_version`. Azure OpenAI fallback models remain in `config.yaml` and are re-activated by reordering provider blocks.

#### Root Cause

| Issue | Detail |
|---|---|
| Wrong LangChain client | `AzureChatOpenAI` is for Azure OpenAI Service; Azure AI Foundry serverless needs `ChatOpenAI` with `base_url` |
| `azure_deployment` vs `model` | `AzureChatOpenAI` uses `azure_deployment`; `ChatOpenAI` uses `model` parameter |
| `api_version` not supported | Azure AI Foundry endpoints are version-free; `api_version: ""` would cause Pydantic `extra="forbid"` rejection |
| `strict=True` incompatibility | OpenAI-only constrained decoding flag; DeepSeek and Llama return 400 API errors when passed |

#### Completed Implementation Steps

| # | File | Change Made | Status |
|---|---|---|---|
| 1 | `infrastructure/llm/types.py` | Added `AZURE_AI_FOUNDRY = "azure_ai_foundry"` to `LLMProvider` enum | ✅ |
| 2 | `infrastructure/llm/config.py` | Added `AzureAIFoundryConfig` (no `api_version`; `get_client_kwargs()` returns `model`+`base_url`); registered in `PROVIDER_CONFIG_REGISTRY`; updated `ProviderConfig` Union | ✅ |
| 3 | `infrastructure/llm/factory.py` | Added `_create_azure_ai_foundry_client()` using `ChatOpenAI`; wired `elif AZURE_AI_FOUNDRY` branch in `_create_client()` | ✅ |
| 4 | `infrastructure/llm/settings.py` | Added `elif AZURE_AI_FOUNDRY` block in `_apply_env_overrides()` mapping `AZURE_AI_FOUNDRY_API_KEY` / `ENDPOINT` | ✅ |
| 5 | `config.yaml` | Added `azure_ai_foundry:` block with `fast`/`smart`; `default` set to `azure_ai_foundry.fast`; removed `api_version: ""`; `embedding` kept under `azure_openai:` | ✅ |
| 6 | `transform_schema_to_text.py`<br>`router.py`<br>`generate_sql.py` | Removed `strict=True` from all three `.with_structured_output()` calls | ✅ |

#### Model Change Summary

| Role | Previous Model | New Model | Provider Type |
|---|---|---|---|
| Fast (routing) | `gpt-4o-mini` via Azure OpenAI | `DeepSeek-V4-Flash` via Azure AI Foundry | `AZURE_AI_FOUNDRY` |
| Smart (SQL gen) | `gpt-4.1` via Azure OpenAI | `Llama-3.3-70B-Instruct` via Azure AI Foundry | `AZURE_AI_FOUNDRY` |
| Embedding | `text-embedding-3-small` via Azure OpenAI | No change | `AZURE_OPENAI` |

---

---

## 19. ⚠️ Performance Issue — Excessive Embedding API Calls & Fix

**File:** `src/ai_agentic_chatbot/agent/subgraphs/sql_query/nodes/retrieve_schemas.py` — `_semantic_search()`

### Observed Behaviour

Every user query triggers **~102 HTTP calls** to the Azure OpenAI embedding endpoint (`text-embedding-3-small`), visible in logs as a burst of identical lines before any SQL is generated:

```
INFO - httpx - HTTP Request: POST https://.../openai/deployments/text-embedding-3-small/embeddings ... "HTTP/1.1 200 OK"
INFO - httpx - HTTP Request: POST https://.../openai/deployments/text-embedding-3-small/embeddings ... "HTTP/1.1 200 OK"
... (repeats ~100 times)
```

---

### Root Cause — Two Systems That Were Never Wired Together

The system has **two parallel schema search mechanisms** that were built independently and never connected:

**1. The `/ingest` pipeline (correct, runs once at setup)**

`VectorSchemaBuilder.build_table_text()` concatenates all schema information — business purpose, key fields, example questions, relationships, operational notes — into one rich text block per table. `PgVectorSchemaStore.ingest()` embeds each block **once** and stores the vectors in pgvector (PostgreSQL). This is the right approach.

**2. `_semantic_search()` at query time (the problem, runs on every request)**

Completely ignores pgvector. Instead reads the YAML file directly, re-embeds every individual field separately with manual weighting, and computes cosine similarity in Python. This runs **102 API calls per user query**.

The developer who wrote `_semantic_search()` wanted weighted multi-level matching (example questions ×2.0, business purpose ×1.5, key fields ×1.2) and implemented it manually — but at the cost of bypassing the already-computed pgvector embeddings entirely.

**Call breakdown per request (4 tables, current `schema_documentation.yaml`):**

```
1 call        → embed_query(user_query)                 # line 98, once before the loop

Per table (×4 tables):
  N calls     → embed_query(each example_question)      # lines 110–116
  1 call      → embed_query(business_purpose)           # line 121
  1 call      → embed_query(search_text)                # line 129
  N calls     → embed_query(each key_field.meaning)     # lines 137–139
```

| Table | `example_questions` | `key_fields` | Per-table calls |
|---|---|---|---|
| `customer` | 8 | 16 | **26** |
| `inventory` | 7 | 13 | **22** |
| `orders` | 6 | 17 | **25** |
| `product` | 7 | 19 | **28** |
| **Total** | | + 1 (user query) | **~102 calls/request** |

The call count grows linearly with `tables × (example_questions + key_fields)`. A richer schema makes it worse.

---

### Impact

| Dimension | Effect |
|---|---|
| **Latency** | 100+ sequential HTTP round-trips before any SQL generation begins, stacking on top of LLM latency |
| **Cost** | Azure OpenAI charges per token. Identical static schema text is re-billed on every user query |
| **Scalability** | Degrades proportionally as more tables or richer schema docs are added |

---

### Solution — Wire `_semantic_search()` to pgvector

The embeddings are already stored in pgvector from `/ingest`. The fix is to use them.

At query time: call `embed_query(user_query)` **once**, then let pgvector do the cosine comparison in SQL against the stored vectors. The rich text block from `VectorSchemaBuilder` already contains all the same information the weighted loop was trying to match separately.

**Before → After:**

```
Before:  ~102 embed_query() calls per user query
After:      1 embed_query() call per user query
```

#### Score Direction — Critical Detail (Verified)

`PGVector.similarity_search_with_score()` returns **cosine distance** (lower = more similar). Applying `score >= 0.3` directly would be backwards — it would pass bad matches and reject good ones.

**Must use `similarity_search_with_relevance_scores()` instead.** Confirmed from `langchain_core/vectorstores/base.py`:

```python
def _cosine_relevance_score_fn(distance: float) -> float:
    return 1.0 - distance   # converts distance → similarity
```

`PGVector` selects this function automatically when using the default `COSINE` distance strategy. `similarity_search_with_relevance_scores()` returns scores in **0–1 where higher = more similar** — directly compatible with the existing `score_threshold=0.3` without any conversion in our code.

---

### Implementation Plan — 4 Tasks, 2 Files

#### Task 1 — Add `search()` to `PgVectorSchemaStore`
**File:** `infrastructure/vector_store/pgvector_store.py`

Add a `search(query, k, score_threshold)` method:
- Calls `self._vectorstore.similarity_search_with_relevance_scores(query, k=k)`
- PGVector handles `embed_query(query)` internally — **1 API call total**
- Filters results by `score >= score_threshold`
- Extracts `doc.metadata["table_name"]` (confirmed present — set by `VectorSchemaBuilder`)
- Returns `List[Tuple[str, float]]` → `(table_name, relevance_score)`

#### Task 2 — Add `get_vector_store()` singleton
**File:** `infrastructure/vector_store/pgvector_store.py`

`PgVectorSchemaStore.__init__` calls `get_engine()` and `get_azure_openai_embedding()` on every construction. Without a singleton, a new DB connection and embedding client would be created on every user query.

```python
_vector_store: Optional[PgVectorSchemaStore] = None

def get_vector_store() -> PgVectorSchemaStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = PgVectorSchemaStore(connection_string="")
    return _vector_store
```

Same pattern as `get_schema_loader()` in `schema_loader.py`.

#### Task 3 — Rewrite `_semantic_search()` in `retrieve_schemas.py`
**File:** `nodes/retrieve_schemas.py`

New flow (return signature `List[Tuple[str, str, float]]` unchanged):

```
1. get_vector_store().search(query, k, score_threshold)
   → 1 API call internally (embed_query on user query only)
   → returns [(table_name, score), ...]

2. get_schema_loader().get_table_docs_for_search()
   → 0 API calls — file read only
   → build lookup dict: {table_name → table_doc}

3. For each (table_name, score) from pgvector:
   → fetch DDL from lookup dict
   → apply router_hint boost ×1.3 (math only, 0 API calls)

4. Return [(table_name, ddl, score), ...]
```

`_expand_related_tables()` is unchanged — it only needs `table_docs` and the retrieved list, no embedding calls.

**Fallback:** If pgvector raises (e.g. `/ingest` was never run, store is empty), fall back to router hints — same as the current `except` clause.

#### Task 4 — Remove dead code from `retrieve_schemas.py`
**File:** `nodes/retrieve_schemas.py`

Remove (all become unused after Task 3):
- `get_azure_openai_embedding` import
- `AzureOpenAIEmbeddingConfig` import
- `AzureOpenAIEmbeddings` import
- `get_embedding` import
- `LLMProvider`, `ModelType` imports
- `_cosine_similarity()` function (lines 192–207)
- Commented-out old `_semantic_search` block (lines 62–81)

Add:
- `from ai_agentic_chatbot.infrastructure.vector_store.pgvector_store import get_vector_store`

---

### Files Changed

| File | Change |
|---|---|
| `infrastructure/vector_store/pgvector_store.py` | Add `search()` method + `get_vector_store()` singleton |
| `nodes/retrieve_schemas.py` | Rewrite `_semantic_search()`, remove dead imports and `_cosine_similarity()` |

**No other files change.** `schema_loader.py`, `ingest_vector_schema.py`, `vector_schema_builder.py`, and `pgvector_store.ingest()` are all untouched.

---

### Pre-condition

`/ingest` must have been called at least once so pgvector has data. If the store is empty, the fallback to router hints activates and a warning is logged. This is not a new constraint — the schema setup pipeline (Section 10.9) already requires `/ingest` to be run during initial setup.

---

### Related Code Locations

| Symbol | File | Line |
|---|---|---|
| `_semantic_search()` — to be rewritten | `nodes/retrieve_schemas.py` | 83 |
| `_cosine_similarity()` — to be removed | `nodes/retrieve_schemas.py` | 192 |
| `PgVectorSchemaStore.ingest()` | `infrastructure/vector_store/pgvector_store.py` | 35 |
| `PgVectorSchemaStore.search()` — to be added | `infrastructure/vector_store/pgvector_store.py` | — |
| `get_vector_store()` — to be added | `infrastructure/vector_store/pgvector_store.py` | — |
| `get_table_docs_for_search()` | `schema_extractor/schema_loader.py` | 53 |
| `VectorSchemaBuilder.build_table_text()` | `schema_extractor/vector_schema_builder.py` | 16 |

---

---

## 20. 🐳 Dockerize & Deploy to Azure VM — Todo List

> **Goal:** Package the FastAPI application into a Docker image and deploy it to an Azure Virtual Machine with TLS, auto-restart, and CI/CD support.

---

### Phase 1 — Pre-Dockerization Cleanup

**1.1 — Fix Windows-specific file paths**

Six env vars in `.env` point to `D:\ai_azure_cert_file\...` — a folder that only exists on the Windows dev machine. On a Linux Docker container that path doesn't exist and the app crashes at startup. The fix differs per variable because they represent three different kinds of files.

#### Category A — Static Prompt Files (bake into the image)

These are read-only `.md` files the LLM uses as system prompts. They don't change at runtime, so they should live inside the repo and ship inside the Docker image like any other source file.

**Affected vars:**
```
ROUTER_PROMPT_PATH=D:\ai_azure_cert_file\chatbot\router_prompts.md
SYSTEM_PROMPT_PATH=D:\ai_azure_cert_file\chatbot\system_prompt.md
```

**Consumed by:**
- `router.py:72` — `open(os.environ["ROUTER_PROMPT_PATH"], "r")`
- `prompt_loader.py:41` — `load_file_content(os.environ["SYSTEM_PROMPT_PATH"])`

**Todos:**
- [ ] Copy `router_prompts.md` and `system_prompt.md` from `D:\ai_azure_cert_file\chatbot\` into the repo at `src/ai_agentic_chatbot/prompts/`
- [ ] Update `.env` to use the container-internal path:
  ```
  ROUTER_PROMPT_PATH=/app/src/ai_agentic_chatbot/prompts/router_prompts.md
  SYSTEM_PROMPT_PATH=/app/src/ai_agentic_chatbot/prompts/system_prompt.md
  ```
- [ ] Delete `CONFIG_DIR=D:\ai_azure_cert_file\chatbot` from `.env` — grep confirms it is never read by any code in `src/`, it is a dead variable

#### Category B — Generated Schema Files (runtime output, bind-mounted volume)

These are **output** files, not inputs. `/schemaText` writes `schema_documentation.yaml`; `generate_schema_summary()` writes `db_schema.json`. They are then read back by the SQL agent and router. They cannot be baked into the image (they depend on the live DB and don't exist at build time). They must survive container restarts, so they need a persistent bind-mount from the VM filesystem.

**Affected vars:**
```
SCHEMA_PATH=D:\ai_azure_cert_file\chatbot\schema_documentation.yaml
SCHEMA_SUMMARY_PATH=D:\ai_azure_cert_file\chatbot\schema_summary\db_schema.json
```

**Consumed by:**
- `schema_loader.py:32` — `Path(os.environ["SCHEMA_PATH"])`
- `transform_schema_to_text.py:86,102,107` — reads and writes both paths
- `schema_loader.py:43` — `os.environ.get("SCHEMA_SUMMARY_PATH")`

**Todos:**
- [ ] Create a `data/` directory at the project root with placeholder files:
  ```
  data/
    .gitkeep
    schema_summary/
      .gitkeep
  ```
- [ ] Add generated files to `.gitignore`:
  ```
  data/*.yaml
  data/**/*.json
  ```
- [ ] Update `.env` to use the container-internal path:
  ```
  SCHEMA_PATH=/app/data/schema_documentation.yaml
  SCHEMA_SUMMARY_PATH=/app/data/schema_summary/db_schema.json
  ```
- [ ] In `docker run` (Phase 7) and `docker-compose.yml` (Phase 3), bind-mount the directory so schema files survive restarts and image upgrades:
  ```
  -v /opt/ai-chatbot/data:/app/data
  ```

#### Category C — MySQL SSL Certificate (commit the public cert into the repo)

`DigiCertGlobalRootG2.crt.pem` is a **public CA root certificate** (not a secret — it is freely downloadable from DigiCert's website). MySQL is not the active datasource (PostgreSQL is the default), but the cert path is wired in `datasource_config.py:81` and would cause a crash if MySQL is ever enabled in a container.

**Affected var:**
```
MYSQL_SSL_CA=D:\ai_azure_cert_file\DigiCertGlobalRootG2.crt.pem
```

**Consumed by:**
- `datasource_config.py:55,65,81` — passed as `ssl_ca` in MySQL connection args

**Todos:**
- [ ] Copy `DigiCertGlobalRootG2.crt.pem` into the repo at `certs/DigiCertGlobalRootG2.crt.pem` (safe to commit — public cert)
- [ ] Update `.env`:
  ```
  MYSQL_SSL_CA=/app/certs/DigiCertGlobalRootG2.crt.pem
  ```

#### Bonus — PostgreSQL env var mismatch bug (`datasource_settings.py`)

The env var override block for PostgreSQL in `datasource_settings.py:86–95` checks for `POSTGRES_HOST`, `POSTGRES_USER`, etc. — but `.env` defines `POSTGRESQL_HOST`, `POSTGRESQL_USER` (with the full `QLQL` suffix). **The override never fires.** The app silently falls back to the credentials hardcoded in `config.yaml` instead of reading from `.env`.

- [ ] Fix `datasource_settings.py` to match the env var names actually used in `.env`:
  ```python
  # Before                          # After
  "POSTGRES_HOST"        →          "POSTGRESQL_HOST"
  "POSTGRES_PORT"        →          "POSTGRESQL_PORT"
  "POSTGRES_DB"          →          "POSTGRESQL_DB"
  "POSTGRES_USER"        →          "POSTGRESQL_USER"
  "POSTGRES_PASSWORD"    →          "POSTGRESQL_PASSWORD"
  ```

#### Summary

| Env Var | Category | Fix |
|---|---|---|
| `ROUTER_PROMPT_PATH` | A — Static prompt | Move `.md` into repo `prompts/`; update path to `/app/src/.../prompts/` |
| `SYSTEM_PROMPT_PATH` | A — Static prompt | Same as above |
| `CONFIG_DIR` | Dead variable | Delete from `.env` |
| `SCHEMA_PATH` | B — Generated output | Create `data/` dir; update path to `/app/data/`; bind-mount on VM |
| `SCHEMA_SUMMARY_PATH` | B — Generated output | Same as above |
| `MYSQL_SSL_CA` | C — Public cert | Commit cert to `certs/`; update path to `/app/certs/` |

**1.2 — Externalize secrets from `config.yaml`**

`config.yaml` has real API keys and database credentials hardcoded inline. Although `config.yaml` is listed in `.gitignore` (so it won't be committed), it **will be copied into the Docker image at build time** unless excluded from the build context. Anyone with access to the image can extract it via `docker inspect` or by running the container. The env var override system already exists in `settings.py` — but it is **not actually being used** because the matching env vars are missing or misnamed in `.env`. The app works today only because secrets are hardcoded in `config.yaml`.

#### Task 1.2.1 — Fix `.env`: add missing env vars for Azure AI Foundry

`settings.py:144–149` overrides foundry credentials using `AZURE_AI_FOUNDRY_API_KEY` and `AZURE_AI_FOUNDRY_ENDPOINT` — but **neither exists in `.env`**. The foundry override never fires. Additionally, `.env` currently stores the Foundry key under the wrong variable name:

```
# Wrong — this is the Foundry key stored under the Azure OpenAI variable name
AZURE_OPENAI_API_KEY=IT1mTuqtvWk...
```

- [ ] Add `AZURE_AI_FOUNDRY_API_KEY=IT1mTuqtvWk...` to `.env`
- [ ] Add `AZURE_AI_FOUNDRY_ENDPOINT=https://dccglobal-ai-services.services.ai.azure.com/openai/v1/` to `.env`
- [ ] Fix `AZURE_OPENAI_API_KEY` in `.env` to hold the actual Azure OpenAI key (`ClNBSEtoqW...` from `config.yaml`), not the Foundry key

#### Task 1.2.2 — Strip all API keys from `config.yaml`

Once `.env` is correct, set all `api_key` fields to `""`. The `settings.py._apply_env_overrides()` fills them at runtime. The keys must remain in the file (Pydantic config classes expect them) — just emptied.

Fields to clear:
- `llm.azure_openai.fast.api_key`
- `llm.azure_openai.smart.api_key`
- `llm.azure_openai.embedding.api_key`
- `llm.azure_ai_foundry.fast.api_key`
- `llm.azure_ai_foundry.smart.api_key`

#### Task 1.2.3 — Strip database credentials from `config.yaml`

`datasource_settings.py` now correctly overrides PostgreSQL credentials from `POSTGRESQL_*` env vars (fixed in 1.1). So `host`, `username`, and `password` in `config.yaml` can be emptied.

Fields to clear:
- `datasources.postgresql.primary.host`
- `datasources.postgresql.primary.username`
- `datasources.postgresql.primary.password`

#### Task 1.2.4 — Remove the dead `embedding:` block from `config.yaml`

`config.yaml` has a top-level `embedding:` section with another hardcoded API key. This block is **never read by any settings class**. `get_azure_openai_embedding()` in `embedding_connection.py:11–16` reads directly from `EMBEDDING_*` env vars, completely bypassing `config.yaml`. The block is dead config that exists only to expose a credential — delete it entirely.

#### Task 1.2.5 — Update `config.example.yaml` to match

`config.example.yaml` is the safe template committed to git. It must mirror the structure of the cleaned `config.yaml` with all secret values set to `""`. The current example is outdated (it still references the old MySQL-first structure).

#### What Does NOT Need to Change

| Item | Reason |
|---|---|
| `EMBEDDING_*` vars in `.env` | `embedding_connection.py` reads directly from env — already correct |
| `POSTGRESQL_*` vars in `.env` | Present and now wired in `datasource_settings.py` after Phase 1.1 fix |
| Non-secret fields in `config.yaml` | Pool sizes, log levels, model names, temperature — safe to keep |

#### Summary

| Task | File | Change |
|---|---|---|
| 1.2.1 | `.env` | Add `AZURE_AI_FOUNDRY_API_KEY`, `AZURE_AI_FOUNDRY_ENDPOINT`; fix `AZURE_OPENAI_API_KEY` |
| 1.2.2 | `config.yaml` | Clear all `api_key` fields in `llm` block |
| 1.2.3 | `config.yaml` | Clear `host`, `username`, `password` in `datasources.postgresql.primary` |
| 1.2.4 | `config.yaml` | Delete entire top-level `embedding:` block |
| 1.2.5 | `config.example.yaml` | Sync structure to match cleaned `config.yaml` |

**1.3 — Audit `.dockerignore`**

The current `.dockerignore` has only 5 entries (`.venv`, `__pycache__`, `.env`, `build`, `worker.egg-info`). Every `docker build` sends the entire project tree to the Docker daemon as build context — including `.git/` history, test files, IDE config, and runtime-generated directories. This makes builds slower, image layers larger, and increases the image attack surface.

#### Task 1.3.1 — Exclude `.git/`
Git history is never needed inside the image and can be hundreds of MB on large repos.
- [ ] Add `.git/`

#### Task 1.3.2 — Make `__pycache__` exclusion recursive
The current `__pycache__` entry only matches the root level. Every `src/` subdirectory has its own cache.
- [ ] Replace `__pycache__` with `**/__pycache__`
- [ ] Add `**/*.pyc` and `**/*.pyo`

#### Task 1.3.3 — Exclude runtime-generated directories
`logs/` and `data/` are written at runtime and will be bind-mounted from the VM filesystem in Phase 7. `temp/` is where `SaveSchemaJson` writes intermediate schema files.
- [ ] Add `logs/`
- [ ] Add `data/`
- [ ] Add `temp/`

#### Task 1.3.4 — Exclude test files
Tests and the pytest cache are dev-only and have no role in the production container.
- [ ] Add `tests/`
- [ ] Add `.pytest_cache/`

#### Task 1.3.5 — Exclude IDE and OS artefacts
- [ ] Add `.idea/`
- [ ] Add `.vscode/`

#### Task 1.3.6 — Exclude log files and top-level docs
`*.log` files belong to the running app, not the image. Top-level markdown docs serve no purpose inside the container.
- [ ] Add `*.log`
- [ ] Add `README.md` and `technical_doc.md` explicitly

> **Important:** Do NOT use `*.md` globally — `src/ai_agentic_chatbot/prompts/*.md` (router and system prompt files added in Phase 1.1) **must ship inside the image**.

#### What stays in the image

| Path | Why it's needed |
|---|---|
| `src/` | Application source code |
| `pyproject.toml` + `poetry.lock` | Dependency installation |
| `certs/` | MySQL SSL certificate (added Phase 1.1) |
| `config.yaml` | Non-secret config (cleaned in Phase 1.2) |
| `config.example.yaml` | Template reference |

**1.4 — Create `logs/` directory placeholder**

Most of this phase requires no code changes — `logging_config.py` was already written correctly for containers.

**Current state (no action needed on these):**
- `logging_config.py:14–15` calls `log_dir.mkdir(exist_ok=True)` at startup — `logs/` auto-creates itself inside the container; no `RUN mkdir` in the Dockerfile is needed
- Log file paths are hardcoded as **relative** strings (`"logs/app.log"`, `"logs/error.log"`, `"logs/datasource.log"`) — these are cross-platform and resolve correctly inside the container when the working directory is `/app`
- The original concern about Windows-hardcoded paths does not apply here — `logging_config.py` was already written portably

**What is missing:**

#### Task 1.4.1 — Add `logs/.gitkeep`
Without a `.gitkeep`, git does not track the `logs/` directory. A fresh clone of the repo will have no `logs/` folder. The auto-create in `logging_config.py` handles runtime, but having the directory visible in the repo is good practice.
- [ ] Create `logs/.gitkeep`
- No `.gitignore` change needed — the existing `*.log` rule already prevents actual log files from being committed

#### Task 1.4.2 — Confirm no further action needed
- [ ] Confirm `logging_config.py:14–15` covers directory creation at runtime inside the container
- [ ] Confirm relative paths (`logs/app.log` etc.) resolve to `/app/logs/*` in the container — no env var override needed

**Phase 7 reminder:** Even though `logging_config.py` auto-creates `logs/` inside the container, log files written there are ephemeral — they disappear when the container is replaced. The bind mount `-v /opt/ai-chatbot/logs:/app/logs` in Phase 7 is what makes logs persist on the VM across restarts and image upgrades.

---

### Phase 2 — Dockerfile

**2.1 — Create `Dockerfile` at project root**

**Key facts from the codebase:**

| Fact | Detail |
|---|---|
| Python version | `>=3.13` (`pyproject.toml:9`) |
| Package manager | Poetry — `pyproject.toml` + `poetry.lock` |
| Package location | `src/ai_agentic_chatbot/` — needs `PYTHONPATH=/app/src` |
| Entry point | `uvicorn ai_agentic_chatbot.server:app --host 0.0.0.0 --port 8000` |
| Runtime dirs | `logs/`, `data/schema_summary/` (auto-created by code, pre-create in image as fallback) |

#### Task 2.1.1 — Base image: `python:3.13-slim`
Debian-based slim variant — matches the `>=3.13` constraint, avoids Alpine's compilation failures with `cryptography` and `psycopg2-binary`.

#### Task 2.1.2 — Install system-level build dependencies
Two packages need OS-level libs at install time:
- `cryptography>=46` needs `gcc` + `libffi-dev` to compile Rust/C extensions
- `psycopg2-binary` needs `libpq-dev` runtime libs on slim images

Install and clean up in a single `RUN` layer to avoid bloating the image:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev libpq-dev \
    && rm -rf /var/lib/apt/lists/*
```

#### Task 2.1.3 — Install Poetry and disable virtualenv creation
Poetry must not create a virtualenv inside the container — packages go directly into system Python:
```dockerfile
RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false
```

#### Task 2.1.4 — Layer-cache-optimised dependency install
Copy `pyproject.toml` and `poetry.lock` **before** source code so that source-only changes don't invalidate the dependency install layer:
```dockerfile
COPY pyproject.toml poetry.lock ./
RUN poetry install --only=main --no-root --no-interaction --no-ansi
```
- `--only=main` excludes the `dev` group (`pytest`, `pytest-asyncio`) from the image
- `--no-root` skips installing the project itself as a package — `PYTHONPATH` handles imports instead

#### Task 2.1.5 — Copy application files
```dockerfile
COPY src/ ./src/
COPY certs/ ./certs/
COPY config.yaml ./
COPY config.example.yaml ./
```
Do **not** copy `.env` — excluded via `.dockerignore`; secrets come from `--env-file` at runtime.

#### Task 2.1.6 — Create runtime directories
`logging_config.py` and `transform_schema_to_text.py` auto-create these at runtime, but pre-creating them ensures correct ownership before the app starts:
```dockerfile
RUN mkdir -p logs data/schema_summary
```

#### Task 2.1.7 — Set `PYTHONPATH`
```dockerfile
ENV PYTHONPATH=/app/src
```

#### Task 2.1.8 — Add a non-root user
Running as `root` inside a container is a security risk. A non-root user limits blast radius if the app is compromised:
```dockerfile
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
RUN chown -R appuser:appgroup /app
USER appuser
```

#### Task 2.1.9 — Expose port and set CMD
Use exec (array) form — `docker stop` sends `SIGTERM` directly to uvicorn for a clean shutdown:
```dockerfile
EXPOSE 8000
CMD ["uvicorn", "ai_agentic_chatbot.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Build order summary
```
FROM python:3.13-slim
  apt-get: gcc, libffi-dev, libpq-dev          ← 2.1.2
  pip install poetry + config virtualenvs=false ← 2.1.3
WORKDIR /app
  COPY pyproject.toml poetry.lock              ← 2.1.4
  poetry install --only=main --no-root         ← 2.1.4
  COPY src/ certs/ config*.yaml                ← 2.1.5
  mkdir -p logs data/schema_summary            ← 2.1.6
  adduser appuser + chown /app                 ← 2.1.8
  USER appuser
ENV PYTHONPATH=/app/src                        ← 2.1.7
EXPOSE 8000                                    ← 2.1.9
CMD ["uvicorn", ...]                           ← 2.1.9
```

**2.2 — Create `entrypoint.sh`**

The entrypoint runs before uvicorn starts and has three jobs: validate required env vars, wait for PostgreSQL to accept connections, then hand off to uvicorn cleanly.

#### Task 2.2.1 — Use `sh` not `bash`, start with `set -e`
Slim Docker images guarantee `/bin/sh` but not `/bin/bash`. `set -e` exits immediately on any error.
- Shebang: `#!/bin/sh`
- First line: `set -e`

#### Task 2.2.2 — Validate required environment variables
Fail fast with a readable message instead of a cryptic Python traceback. Split into groups matching each subsystem:

| Group | Variables |
|---|---|
| PostgreSQL | `POSTGRESQL_HOST`, `POSTGRESQL_PORT`, `POSTGRESQL_DB`, `POSTGRESQL_USER`, `POSTGRESQL_PASSWORD` |
| Active LLM (Foundry) | `AZURE_AI_FOUNDRY_API_KEY`, `AZURE_AI_FOUNDRY_ENDPOINT` |
| Embeddings | `EMBEDDING_API_KEY`, `EMBEDDING_ENDPOINT`, `EMBEDDING_MODEL_NAME`, `EMBEDDING_API_VERSION` |
| Prompt files | `ROUTER_PROMPT_PATH`, `SYSTEM_PROMPT_PATH` |

- Write a `check_var()` helper that prints `ERROR: '<VAR>' is not set` and exits 1 if empty
- Call it for all 13 required vars

#### Task 2.2.3 — Wait for PostgreSQL TCP readiness
`server.py` catches datasource init failures silently (`lifespan` except block, line 55) — the app starts but is broken. A wait loop ensures PostgreSQL is reachable before handing off. Use a Python one-liner (Python is already in the image — no extra packages):

```sh
python -c "
import socket, time, sys
host, port = '$POSTGRESQL_HOST', int('$POSTGRESQL_PORT')
for i in range(30):
    try:
        socket.create_connection((host, port), timeout=2).close()
        print('PostgreSQL is ready.')
        sys.exit(0)
    except OSError:
        print(f'Attempt {i+1}/30 — not ready, retrying in 2s...')
        time.sleep(2)
print('ERROR: PostgreSQL did not become ready after 60s.')
sys.exit(1)
"
```
30 attempts × 2 seconds = 60 second max wait. Exits 1 if never ready so the container fails visibly.

#### Task 2.2.4 — Hand off to CMD with `exec "$@"`
`exec "$@"` replaces the shell process with the CMD passed at runtime. This means `SIGTERM` from `docker stop` goes directly to uvicorn for graceful shutdown. It also keeps the entrypoint flexible — CMD can be overridden at `docker run` time without touching this script.
- Final line: `exec "$@"`

#### Task 2.2.5 — Wire into the Dockerfile
The Dockerfile must be updated to split the existing `CMD` into `ENTRYPOINT + CMD`:
```dockerfile
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "ai_agentic_chatbot.server:app", "--host", "0.0.0.0", "--port", "8000"]
```
`ENTRYPOINT` always runs the validation and wait. `CMD` is the default command passed to `exec "$@"` and can be overridden at runtime.

#### Script structure
```
#!/bin/sh
set -e
  check_var() helper
  Validate PostgreSQL vars        (5 vars)
  Validate LLM vars               (2 vars)
  Validate Embedding vars         (4 vars)
  Validate Prompt path vars       (2 vars)
  Wait loop — PostgreSQL TCP      (30 × 2s, Python one-liner)
exec "$@"   ← hands off to CMD (uvicorn)
```

---

### Phase 3 — Docker Compose (Local Testing)

**3.1 — Create `docker-compose.yml`**

Both local testing and production connect to the real Azure PostgreSQL. All connection details come from `.env`. The compose file is intentionally minimal — one service, three volume mounts, all config via `env_file`.

> **Design decision (2026-06-08):** A local pgvector sidecar was considered and removed. Running against the real Azure database gives accurate test results and avoids maintaining separate local seed data. The `POSTGRESQL_SSLMODE` env override added to `datasource_settings.py` remains useful for any future local container use.

#### Task 3.1.1 — Add `POSTGRESQL_SSLMODE` env override to `datasource_settings.py`
Added for flexibility — `PostgreSQLConfig.sslmode` previously had no env override:
```python
if "POSTGRESQL_SSLMODE" in os.environ:
    ds_data["sslmode"] = os.environ["POSTGRESQL_SSLMODE"]
```

#### Task 3.1.2 — Define the `app` service
- `build: .` — builds from `Dockerfile` at project root
- `env_file: .env` — all connection details (PostgreSQL, LLM, embeddings) from `.env`
- Port `8000:8000`
- Three volume mounts:
  - `./config.yaml:/app/config.yaml:ro` — non-secret config, read-only
  - `./logs:/app/logs` — persist logs on the host
  - `./data:/app/data` — persist generated schema files

#### Compose structure
```
services:
  app:
    build: .
    ports: 8000:8000
    env_file: .env
    volumes: config.yaml(ro), logs/, data/
```

**3.2 — Create `.env.docker` template**

A safe-to-commit file mirroring `.env` structure — all secret values stripped, every variable annotated as required or optional, with notes on which vars are overridden by `docker-compose.yml` for local runs.

- [ ] **3.2.1** — Strip all secret values (`API_KEY`, `PASSWORD`); keep safe non-secret defaults (endpoints, versions, paths, timeouts)
- [ ] **3.2.2** — Annotate each variable as `# REQUIRED` or `# OPTIONAL`
- [ ] **3.2.3** — Mark PostgreSQL connection vars and `POSTGRESQL_SSLMODE` as `# overridden by docker-compose.yml for local` — they don't need filling for local compose runs
- [ ] **3.2.4** — Add a header comment explaining the file is a template to copy to `.env` and fill in before running the container
- [ ] **3.2.5** — Commit `.env.docker` to git; `.env` remains gitignored

---

### Phase 4 — Azure VM Setup

**4.1 — Provision the Azure VM**
- Recommended size: `Standard_B2s` or `Standard_D2s_v3` (2 vCPU, 4–8 GB RAM)
- OS: Ubuntu Server 22.04 LTS
- Open inbound ports: `22` (SSH), `8000` (API), `443` (HTTPS via reverse proxy)
- Assign a static public IP or DNS label
- Add the VM to the same VNet as the Azure PostgreSQL instance, or configure the PostgreSQL firewall to allow the VM's IP

**4.2 — Install Docker on the VM**
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
# Add Docker's official GPG key and repo, then:
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER
```

**4.3 — Configure Azure PostgreSQL firewall**
- Add the VM's outbound public IP to the Azure PostgreSQL `Allowed IPs` list, **or**
- Place the VM in the same VNet and use a private endpoint (recommended for production)

---

### Phase 5 — Azure Container Registry

**5.1 — Create an Azure Container Registry (ACR)**
```bash
az acr create --name <registry-name> --resource-group <rg> --sku Basic
```
- Enable admin access, or use managed identity for the VM

The note "Enable admin access, **or** use managed identity for the VM" refers to **how the VM authenticates to pull your image from the ACR**. Two options:

#### Option A — Enable Admin Access (Simple, but less secure)

ACR has a built-in admin user you can enable. It gives you a username + two passwords.

```bash
az acr update --name <registry-name> --admin-enabled true
```

Then on the VM, log in using those credentials:

```bash
az acr login --name <registry-name>
# OR
docker login <registry-name>.azurecr.io \
  --username <registry-name> \
  --password <password-from-portal>
```

**Drawback:** Static password — if it leaks, anyone can pull your images. Requires manual credential rotation.

#### Option B — Managed Identity (Recommended for production)

The VM's Azure-managed identity is granted `AcrPull` permission directly. No passwords stored anywhere.

**Step 1:** Enable system-assigned managed identity on the VM:
```bash
az vm identity assign \
  --resource-group <rg> \
  --name <vm-name>
```

**Step 2:** Get the VM's principal ID:
```bash
az vm show \
  --resource-group <rg> \
  --name <vm-name> \
  --query identity.principalId \
  --output tsv
```

**Step 3:** Get the ACR's resource ID:
```bash
az acr show --name <registry-name> --query id --output tsv
```

**Step 4:** Assign the `AcrPull` role — VM identity → ACR:
```bash
az role assignment create \
  --assignee <principal-id-from-step-2> \
  --role AcrPull \
  --scope <acr-resource-id-from-step-3>
```

**Step 5:** On the VM, log in without a password (Azure uses the VM's identity token automatically):
```bash
az acr login --name <registry-name>
```

| | Admin Access | Managed Identity |
|---|---|---|
| Setup effort | Minimal | ~5 commands |
| Security | Weaker (static password) | Strong (no secrets) |
| Suitable for | Quick testing | Production |

**Recommendation:** Use Managed Identity for production. Admin access is acceptable for initial pipeline testing only.

**5.2 — Build and push the image**
```bash
az acr login --name <registry-name>
docker build -t <registry-name>.azurecr.io/ai-agentic-chatbot:latest .
docker push <registry-name>.azurecr.io/ai-agentic-chatbot:latest
```

**5.3 — Configure the VM to pull from ACR**
- Run `az acr login` on the VM, or assign the `AcrPull` role to the VM's managed identity

---

### Phase 6 — Secrets Management on the VM

**6.1 — Choose a secrets strategy**
- **Option A (simple):** Place `.env` at `/opt/ai-chatbot/.env` on the VM, restrict permissions to `600`, reference it in the `docker run` command
- **Option B (recommended):** Use **Azure Key Vault** — store all API keys there, use the VM's managed identity to fetch secrets at startup via a small init script

**Option A — Solution Steps**

> **Note:** `scp` runs as your regular SSH user and cannot write directly into `/opt/ai-chatbot/` (owned by root). Copy to home directory first, then move with `sudo`.

```bash
# Step 1: Copy .env to your home directory on the VM
scp .env <user>@<vm-ip>:~/ai-chatbot.env

# Step 2: SSH into the VM, then move it to the target location
sudo mv ~/ai-chatbot.env /opt/ai-chatbot/.env

# Step 3: Lock down permissions — owned by your SSH user, not root
# Replace <user> with your actual VM username (e.g. azureuser)
sudo chown <user>:<user> /opt/ai-chatbot/.env
sudo chmod 600 /opt/ai-chatbot/.env
```

Reference the `.env` file when running the container:
```bash
docker run --env-file /opt/ai-chatbot/.env ...
```

**6.2 — Mount `config.yaml` from the VM filesystem**
- Store `config.yaml` (non-secret parts only) at `/opt/ai-chatbot/config.yaml` on the VM
- Bind-mount it read-only: `-v /opt/ai-chatbot/config.yaml:/app/config.yaml:ro`

> **Note:** `config.yaml` is intentionally **not baked into the Docker image** — it is mounted from the VM at runtime so model settings and tuning knobs can be changed without rebuilding the image.

**Solution Steps**

```bash
# Step 1: Copy config.yaml to your home directory on the VM
scp config.yaml <user>@<vm-ip>:~/ai-chatbot-config.yaml

# Step 2: SSH into the VM, then move it to the target location
sudo mv ~/ai-chatbot-config.yaml /opt/ai-chatbot/config.yaml

# Step 3: Set permissions (readable by all, not secret)
sudo chmod 644 /opt/ai-chatbot/config.yaml
sudo chown root:root /opt/ai-chatbot/config.yaml
```

Add the bind-mount flag when running the container:
```bash
docker run \
  --env-file /opt/ai-chatbot/.env \
  -v /opt/ai-chatbot/config.yaml:/app/config.yaml:ro \
  ...
```

---

### Phase 7 — Container Deployment on the VM

**7.1 — Pull and run the container**

Use `docker-compose.prod.yml` (checked into the repo) to manage the container on the VM. It pulls the pre-built image from ACR, mounts secrets and config from the VM filesystem, and uses named volumes for `logs` and `temp`.

#### Step 1 — Rebuild and push the image (local machine)

> **Important — Windows line endings (CRLF) issue:** `entrypoint.sh` edited on Windows has `\r\n` line endings. Linux cannot execute it because the shebang becomes `#!/bin/sh\r` — an invalid path. The `Dockerfile` strips `\r` via `sed` before `chmod +x`, and `.gitattributes` enforces `eol=lf` for `.sh` files going forward. Always rebuild the image after any change to `entrypoint.sh`.

```bash
# On your local Windows machine:
docker build -t dccglobalregistry.azurecr.io/ai-agentic-chatbot:latest .
docker push dccglobalregistry.azurecr.io/ai-agentic-chatbot:latest
```

#### Step 2 — Copy `docker-compose.prod.yml` to the VM

Azure VMs use key-based SSH auth (password auth is disabled by default). `scp` requires the `-i` flag pointing to your private key:

```bash
scp -i ~/.ssh/id_rsa docker-compose.prod.yml <user>@<vm-ip>:~/docker-compose.prod.yml
```

If `scp` is not available or auth keeps failing, SSH into the VM first and paste the file contents directly:

```bash
ssh -i ~/.ssh/id_rsa <user>@<vm-ip>

# On the VM — create the file by pasting:
cat > ~/docker-compose.prod.yml << 'EOF'
services:
  app:
    image: dccglobalregistry.azurecr.io/ai-agentic-chatbot:latest
    restart: unless-stopped
    ports:
      - "8000:8000"
    env_file:
      - /opt/ai-chatbot/.env
    volumes:
      - /opt/ai-chatbot/config.yaml:/app/config.yaml:ro
      - ai-chatbot-logs:/app/logs
      - ai-chatbot-temp:/app/temp

volumes:
  ai-chatbot-logs:
  ai-chatbot-temp:
EOF
```

#### Step 3 — Login to ACR and pull the image (on the VM)

```bash
# Option A — admin credentials
docker login dccglobalregistry.azurecr.io

# Option B — Azure CLI (managed identity)
az acr login --name dccglobalregistry
```

```bash
docker pull dccglobalregistry.azurecr.io/ai-agentic-chatbot:latest
```

#### Step 4 — Start the container

```bash
# Start (detached)
docker compose -f ~/docker-compose.prod.yml up -d

# Stop
docker compose -f ~/docker-compose.prod.yml down

# Restart after a new image push
docker compose -f ~/docker-compose.prod.yml pull
docker compose -f ~/docker-compose.prod.yml up -d

# Tail live logs
docker compose -f ~/docker-compose.prod.yml logs -f
```

> **Note:** `logs` and `temp` use **named Docker volumes** (not bind mounts). The container runs as a non-root `appuser` — bind-mounting a host directory owned by root causes a `PermissionError` when the app tries to write log files. Named volumes are managed by Docker and are writable by the container user automatically.
>
> To inspect logs:
> ```bash
> docker compose -f ~/docker-compose.prod.yml logs -f       # stdout/stderr
> docker exec ai-agentic-chatbot-app-1 cat /app/logs/app.log  # file logs
> ```

#### Troubleshooting — wrong Azure OpenAI endpoint being used

**Symptom:** `/schemaText` (or any endpoint) hits the old URL `https://dcc-azure-openai.cognitiveservices.azure.com/...` even though `config.yaml` has the correct URL `https://dccglobal-ai-services.openai.azure.com/`.

**Cause:** `settings.py` applies env var overrides on top of `config.yaml` at startup. If `AZURE_OPENAI_ENDPOINT` in `/opt/ai-chatbot/.env` on the VM still holds the old URL, it wins over the config file value every time.

**Fix:** Update the env file on the VM and restart:

```bash
# 1. Edit .env on the VM
nano /opt/ai-chatbot/.env
# Set: AZURE_OPENAI_ENDPOINT=https://dccglobal-ai-services.openai.azure.com/

# 2. Restart the container to pick up the change
docker compose -f ~/docker-compose.prod.yml down
docker compose -f ~/docker-compose.prod.yml up -d
```

> **Rule:** `config.yaml` is the source of truth for non-secret settings, but any env var in `.env` silently overrides it. When the app hits an unexpected endpoint, check `/opt/ai-chatbot/.env` on the VM first.

**7.2 — Verify the deployment**

From inside the VM:
```bash
docker compose -f docker-compose.prod.yml logs
curl http://localhost:8000/health
curl http://localhost:8000/db-health
```

To verify from outside the VM (browser or curl using the VM's public IP), port `8000` must be opened in the Azure NSG. Do this **temporarily for testing only** — close it again once Phase 8 (Nginx + HTTPS) is in place.

**Enable external access on port 8000 (testing only):**
```bash
az network nsg rule create \
  --resource-group <your-resource-group> \
  --nsg-name <your-nsg-name> \
  --name allow-chatbot-8000 \
  --protocol Tcp \
  --direction Inbound \
  --priority 1000 \
  --destination-port-range 8000 \
  --access Allow
```

Then access via:
```
http://<vm-public-ip>:8000/health
http://<vm-public-ip>:8000/db-health
```

**Disable external access on port 8000 (after testing):**
```bash
az network nsg rule delete \
  --resource-group <your-resource-group> \
  --nsg-name <your-nsg-name> \
  --name allow-chatbot-8000
```

> **Note:** Port 8000 should not remain open in production — it exposes the app over plain HTTP. Phase 8 (Nginx reverse proxy + HTTPS) handles public traffic on port `443` and keeps port `8000` closed externally.

---

### Phase 8 — Reverse Proxy & HTTPS

**8.1 — Install Nginx on the VM**
- Acts as a reverse proxy in front of the container on port `8000`
- Handles TLS termination

**8.2 — Obtain a TLS certificate**
- Use Let's Encrypt via Certbot (if a domain name points to the VM), **or**
- Use Azure Application Gateway in front of the VM for managed TLS

**8.3 — Configure Nginx**
```nginx
server {
    listen 443 ssl;
    server_name <your-domain>;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;   # Required for SSE /stream endpoint
    }
}
```

> ⚠️ `proxy_buffering off` is critical — the `/stream` endpoint uses **Server-Sent Events** and will hang with buffering enabled.

---

### Phase 9 — CI/CD Pipeline (Optional)

**9.1 — Create a GitHub Actions workflow**
- Trigger on push to `main`
- Steps: checkout → build Docker image → push to ACR → SSH into Azure VM → pull new image → restart container

**9.2 — Store secrets in GitHub Actions**
- `ACR_USERNAME`, `ACR_PASSWORD`, `VM_SSH_KEY`, `VM_HOST`

---

### Phase 10 — Observability & Maintenance

**10.1 — Set up log forwarding**
- Bind-mount `/app/logs` to the VM filesystem (already covered in Phase 7.1)
- Optionally ship logs to Azure Monitor / Log Analytics via the Azure Monitor Agent

**10.2 — Container auto-restart**
- `--restart unless-stopped` ensures the container comes back after VM reboots (set in Phase 7.1)

**10.3 — Health check cron on the VM**
```bash
# /etc/cron.d/ai-chatbot-healthcheck
*/5 * * * * root curl -sf http://localhost:8000/health || docker restart ai-chatbot
```

**10.4 — First-run schema ingestion**
- After first deployment, call the schema setup pipeline once (see [Section 10.9](#109-full-schema-setup-pipeline-3-step)):
  1. `GET /schemaJson`
  2. `GET /schemaText`
  3. `GET /ingest`
- This populates pgvector so the SQL agent can perform semantic table search

---

### Files to Create / Modify

| File | Action | Notes |
|---|---|---|
| `Dockerfile` | **Create** | Python 3.13-slim, Poetry, expose 8000 |
| `entrypoint.sh` | **Create** | Env var validation + uvicorn exec |
| `docker-compose.yml` | **Create** | App + pgvector services for local testing |
| `.env.docker` | **Create** | Safe template — no real values |
| `.dockerignore` | **Update** | Add logs/, .git, tests/, config.yaml |
| `config.yaml` | **Update** | Remove hardcoded secrets — use env var references |
| `.env` | **Update** | Replace Windows paths with Linux-compatible paths |
| `.github/workflows/deploy.yml` | **Create** | CI/CD pipeline (optional) |

> **Critical blockers to fix first:** Item 1.1 (Windows paths) and 1.2 (secrets in `config.yaml`) — the container will fail to start if these are not resolved before building the image.

---

---

## 21. 🎯 NL-to-SQL Accuracy Improvements — Improvement Plan

> **Analysis date:** 2026-06-10 | **GitHub Issue:** [#15](https://github.com/dccservicespl/ai-agentic-chatbot/issues/15)
> **Context:** Fresh audit comparing `instruction.txt` prescriptions, `system_prompt.md`, and the live codebase. Four bugs and missing database objects identified.

---

### What `instruction.txt` Prescribes

The instruction file defines four infrastructure items that form the foundation of accurate NL-to-SQL generation:

| Object | Type | Purpose |
|---|---|---|
| `v_sales_summary` | VIEW | Flat denormalized JOIN of all 5 tables — AI generates simpler SQL against one view instead of complex multi-table JOINs |
| `schema_metadata` | TABLE + inserts | Human-readable descriptions of every table and column with sample values and join key flags |
| `business_glossary` | TABLE + inserts | Maps plain-English user terms ("revenue", "low stock", "top customers") to exact SQL expressions |
| 12 indexes | INDEX | Composite indexes on customer, orders, product, inventory — covers the most common NL filter columns |

---

### Current State of `system_prompt.md`

The system prompt is already comprehensive and well-structured. It contains:

- Full column reference for all 5 tables
- `v_sales_summary` view columns documented (all 43 columns listed)
- 30+ business definitions (revenue, stock level, order lifecycle terms, date ranges)
- The 8-stage order lifecycle (PENDING → COMPLETED / VOID)
- 12 SQL generation rules (ILIKE, COALESCE, LIMIT 100, COUNT DISTINCT, etc.)
- Intent classification (SQL_QUERY / GREETING / EXPLAIN / OUT_OF_SCOPE / UNKNOWN)
- 30+ example query pairs across all domains

**The prompt file is in good shape. The gaps are in the code and database — not the prompt.**

---

### Bugs Found

#### BUG 1 — `system_prompt.md` never reaches the SQL generator (Critical)

**File:** `agent/router.py:79` — correctly loads `get_system_prompt()` and passes it to the router LLM.

**File:** `agent/subgraphs/sql_query/nodes/generate_sql.py:_create_generation_prompt()` — uses a **fully hardcoded inline prompt** with no reference to `system_prompt.md`. The SQL generator is blind to:

- `v_sales_summary` preference rule (Rule 8)
- All 30+ business definitions (revenue, stock level, etc.)
- All 12 SQL rules (ILIKE, COALESCE, LIMIT 100, COUNT DISTINCT, etc.)
- The order lifecycle
- All 30+ example query patterns

**Impact:** The SMART LLM (Llama-3.3-70B) generates SQL without any domain knowledge, business context, or the rules defined in the system prompt.

**Fix:** Call `get_system_prompt()` inside `_create_generation_prompt()` and prepend it to the prompt.

---

#### BUG 2 — `SchemaExtractor` never extracts views (Critical)

**File:** `schema_extractor/SchemaExtractor.py:extract_database_schema()`

Uses only `inspector.get_table_names()`. PostgreSQL views require `inspector.get_view_names()` — a separate call. Even after `v_sales_summary` is created in the database, it will **never appear** in `db_schema.json`, `schema_documentation.yaml`, or the pgvector store. The view is invisible to the semantic retrieval step.

**Fix:** Add `_extract_view_schema()` using `inspector.get_view_names()` and `inspector.get_view_definition()`, included in `extract_database_schema()` output.

---

#### BUG 3 — Router schema_summary iteration is broken

**File:** `agent/router.py:76-78`

```python
schema_summary = schema_loader.load_schema_summary()   # returns full JSON dict
schema_summary_text = "\n".join(
    [f"- {table}: {desc}" for table, desc in schema_summary.items()]
)
```

`schema_summary` structure (from `generate_schema_summary()` in `transform_schema_to_text.py`):

```json
{
  "database_name": "...",
  "version": "v1",
  "tables": [{ "table": "...", "bussiness_purpose": "...", "example_questions": [...] }]
}
```

Calling `.items()` on this dict yields the top-level keys `database_name`, `version`, `tables` as "table names". The router prompt receives `"- database_name: ai_chatbot_db"` instead of actual table names and purposes.

**Fix:** Replace `.items()` iteration with `schema_summary.get("tables", [])` loop using `t["table"]` and `t["bussiness_purpose"]`.

---

#### BUG 4 — LIMIT inconsistency: 10 rows in code vs 100 in system prompt

**File:** `agent/subgraphs/sql_query/nodes/generate_sql.py:117`

```
"Limit results to a reasonable number (max 10 rows)"
```

**system_prompt.md Rule 5:**

```
Default LIMIT 100 — unless user says "all", "every", or specifies a number
```

The SQL generator applies a 10-row cap while the system prompt promises 100.

**Fix:** Update `generate_sql.py:117` to align with the system prompt default.

---

### Missing Database Objects

None of the following have a migration script in the repository:

| Object | Status | Impact if missing |
|---|---|---|
| `v_sales_summary` view | ❌ not created | AI generates complex multi-table JOINs instead of querying the optimized view |
| `schema_metadata` table + inserts | ❌ not created | Column-level business descriptions unavailable for dynamic injection |
| `business_glossary` table + inserts | ❌ not created | Term → SQL mapping not available at runtime from DB |
| 12 indexes on customer/orders/product/inventory | ❌ not created | NL queries filtering by date, customer, product run without index support |

---

### TODO List (Prioritized)

#### P0 — Database objects (prerequisite for everything) ✅

- [x] Run all SQL from `instruction.txt` in the PostgreSQL business database:
  - `CREATE OR REPLACE VIEW v_sales_summary AS ...` (joins all 5 tables)
  - `CREATE TABLE schema_metadata (...)` + all `INSERT INTO schema_metadata` rows
  - `CREATE TABLE business_glossary (...)` + all `INSERT INTO business_glossary` rows
  - All 12 `CREATE INDEX` statements on customer, orders, product, inventory
- [ ] Add a `migrations/` or `scripts/` folder in the repo with these DDL statements so setup is reproducible

#### P1 — Fix SQL generator not using `system_prompt.md`

- [ ] **`generate_sql.py`:** Call `get_system_prompt()` and inject it at the start of `_create_generation_prompt()` so the SMART LLM knows the view preference, business definitions, SQL rules, and example patterns

#### P2 — Fix `SchemaExtractor` missing views

- [ ] **`SchemaExtractor.py`:** Add `_extract_view_schema()` using `inspector.get_view_names()` + `inspector.get_view_definition()`; include views in `extract_database_schema()` output
- [ ] **`server.py` (schemaJson config):** Add `schema_metadata` and `business_glossary` to `exclude_tables` in `SchemaExtractionConfig` — these are internal metadata tables the AI should not query directly

#### P3 — Fix router schema_summary bug

- [ ] **`router.py:76-78`:** Replace `schema_summary.items()` with `schema_summary.get("tables", [])` loop using `t["table"]` and `t["bussiness_purpose"]`

#### P4 — Fix LIMIT inconsistency ✅ (resolved by P1)

- [x] **`generate_sql.py:117`:** Hardcoded `"max 10 rows"` removed — P1 rewrote `_create_generation_prompt()` to inject `system_prompt.md` which already contains Rule 5 ("Default LIMIT 100 unless user specifies otherwise"). No separate fix needed.

#### P5 — Re-ingest vector store after P0 + P2

- [ ] After creating the view and fixing the extractor, re-run the full schema setup pipeline:
  1. `GET /schemaJson` — extracts tables + `v_sales_summary` view into `db_schema.json`
  2. `GET /schemaText` — LLM enriches schema into `schema_documentation.yaml`
  3. `GET /ingest?force_reset=true` — clears stale rows, embeds and upserts all table/view chunks

---

### P5 — Deep Dive: Re-ingest Vector Store

#### Why This Step Exists

After P0 creates `v_sales_summary` in the database and P2 fixes `SchemaExtractor` to call `get_view_names()`, the pgvector store is still stale — it was ingested before the view existed and has no embedding for it. Until P5 is run, the SQL agent's semantic table discovery (`retrieve_schemas` node) will never surface `v_sales_summary`, so the LLM will keep generating complex multi-table JOINs instead of querying the flat view.

#### The 3-Step Pipeline

**Step 1 — `GET /schemaJson`**

Calls `SchemaExtractor.extract_database_schema()` → now also calls `get_view_names()` + `_extract_view_schema()` (P2 fix). Writes `temp/db_schema.json` containing all 5 tables + the view.

**Step 2 — `GET /schemaText`**

Reads `db_schema.json`, sends each object to the LLM, generates `schema_documentation.yaml` — the enriched description (business purpose, key fields, example questions) that will be embedded.

**Step 3 — `GET /ingest?force_reset=true`**

`force_reset=true` calls `get_vector_store().reset_collection()` first — drops + recreates the pgvector collection to remove any duplicate rows from previous ingest runs. Then upserts 6 clean chunks (5 tables + 1 view) using stable uuid5 IDs.

#### Dependency

P5 cannot run until P0 creates `v_sales_summary` in the database. Order: **P0 → P5**.

---

#### P6 — Runtime glossary injection (enhancement)

- [x] Add `glossary_lookup.py` with two functions:
  - `fetch_glossary_hints(user_query, engine)` — SQL `ILIKE` match of user query against `business_glossary.term`, returns formatted hint block
  - `fetch_column_hints(table_names, engine)` — queries `schema_metadata` for column descriptions of retrieved tables, returns formatted hint block
- [x] Update `generate_sql_node` to call both functions and inject hints into `_create_generation_prompt()`

---

### P6 — Deep Dive: Runtime Glossary Injection

#### Problem

The SQL generation LLM has no reliable way to map business language to SQL constructs. When a user says *"show me revenue from active customers this month"*, the LLM must guess:
- `revenue` → `SUM(quantity * unit_price)` or `grand_total`?
- `active` → `activation_status = 'ACTIVE'`?
- `this month` → `WHERE order_date >= date_trunc('month', CURRENT_DATE)`?

Wrong guesses produce syntactically valid but semantically wrong SQL.

#### The Two Lookup Functions

**`fetch_glossary_hints(user_query, engine)`** — `business_glossary` lookup

Runs a single SQL query that matches any `term` from `business_glossary` whose text appears (case-insensitive) in the user's query:

```sql
SELECT term, sql_meaning
FROM business_glossary
WHERE $user_query ILIKE '%' || term || '%'
```

Returns a formatted block injected into the prompt:

```
## BUSINESS GLOSSARY (matched to your query)
- "revenue" → SUM(sales.quantity * sales.unit_price)
- "active customers" → customers WHERE activation_status = 'ACTIVE'
- "this month" → WHERE order_date >= date_trunc('month', CURRENT_DATE)
```

**`fetch_column_hints(table_names, engine)`** — `schema_metadata` lookup

Queries column-level descriptions for the tables the pgvector search returned as relevant:

```sql
SELECT table_name, column_name, data_type, description, sample_values, is_join_key
FROM schema_metadata
WHERE table_name = ANY(:tables) AND column_name IS NOT NULL
ORDER BY table_name, column_name
```

Returns a formatted block:

```
## COLUMN HINTS (for retrieved tables)
- orders.order_no [VARCHAR, join key]: Unique order number. Referenced by sales.reference_no.
- orders.grand_total [NUMERIC]: Final order total after discount and delivery fees.
- sales.reference_no [VARCHAR, join key]: Links sales line to an order via orders.order_no.
```

#### Where They Are Injected

Both blocks are appended inside `_create_generation_prompt()` in `generate_sql.py`, between the retrieved schema DDL and the user request:

```
{system_prompt.md content}
---
## RETRIEVED SCHEMA CONTEXT
{DDL from pgvector}

{BUSINESS GLOSSARY block — if any matches}

{COLUMN HINTS block — if any tables matched}

## USER REQUEST
{user_query}
```

If the tables don't exist yet (P0 not run), both functions catch the `ProgrammingError` and return an empty string — the prompt degrades gracefully with no crash.

#### Files Changed

| File | Change |
|---|---|
| `agent/subgraphs/sql_query/nodes/glossary_lookup.py` | New file — `fetch_glossary_hints()` and `fetch_column_hints()` |
| `agent/subgraphs/sql_query/nodes/generate_sql.py` | Import and call both lookup functions; pass `engine` + `table_names` to `_create_generation_prompt()` |

---

### Status Summary

| Item | `system_prompt.md` | Code | Database |
|---|---|---|---|
| `v_sales_summary` columns documented | ✅ all 43 columns listed | ❌ not extracted by `SchemaExtractor` | ❌ view may not exist |
| Business glossary | ✅ 30+ terms inline | ❌ not injected into SQL gen | ❌ table may not exist |
| SQL rules (ILIKE, LIMIT, COALESCE) | ✅ 12 rules defined | ❌ SQL gen uses inline hardcoded prompt | — |
| 12 indexes | — | — | ❌ not created |
| `schema_metadata` | — | ❌ not used anywhere | ❌ not created |
| Router schema summary | ✅ feeds router intent | ❌ BUG — `.items()` gives wrong keys | — |
| `system_prompt.md` in router | ✅ | ✅ `router.py:79` | — |
| `system_prompt.md` in SQL gen | ✅ | ❌ `generate_sql.py` ignores it | — |

**Highest-leverage fixes:** P1 (inject system prompt into SQL generator) + P0 (create DB objects). Together these will immediately improve SQL accuracy and view utilisation.

---

### P2 — Deep Dive: Fix `SchemaExtractor` to Extract Views

#### The Core Problem

The schema setup pipeline runs in 3 steps:

```
GET /schemaJson  →  GET /schemaText  →  GET /ingest
  (extract)            (enrich)         (embed into pgvector)
```

**Step 1 (`/schemaJson`)** calls `SchemaExtractor.extract_database_schema()` which uses SQLAlchemy's inspector. The inspector has **two separate methods** for tables and views — but only one is ever called:

```python
inspector.get_table_names()   # called  → ['customer', 'orders', 'sales', 'product', 'inventory']
inspector.get_view_names()    # never called → ['v_sales_summary']  ❌
```

Because `get_view_names()` is never called, `v_sales_summary` is completely invisible to the entire pipeline. It never enters `db_schema.json`, never gets enriched into YAML, and never gets embedded into pgvector.

The result: even though P1 (system prompt injection) tells the SQL generator to *prefer* `v_sales_summary`, when the semantic retrieval step searches for relevant tables, there is no pgvector entry for the view — it can never be retrieved and passed to the LLM.

---

#### What Needs to Change — 2 Things

**Thing 1 — Extract `v_sales_summary` as a view**

Add a view extraction loop alongside the existing table loop in `SchemaExtractor.extract_database_schema()`. For each view, the inspector's `get_columns()` call works identically to tables in PostgreSQL, so column extraction is free. `get_view_definition()` returns the raw `SELECT ... FROM ...` DDL that defines the view.

The DDL stored in pgvector for `v_sales_summary` will look like:

```sql
CREATE OR REPLACE VIEW public.v_sales_summary AS
SELECT s.id AS sales_id, s.order_date, s.quantity * s.unit_price AS line_total,
       c.customer_name, c.code AS customer_code, c.activation_status AS customer_status,
       o.order_no, o.order_status, o.grand_total,
       p.product_name, p.product_code, p.status AS product_status,
       i.on_hand, i.available_quantity, i.expected_quantity, ...
FROM sales s
LEFT JOIN customer c ON s.customer_code = c.code
LEFT JOIN orders o ON s.reference_no = o.order_no
LEFT JOIN product p ON s.product_code = p.product_code
LEFT JOIN inventory i ON s.product_code = i.product_code
```

Once this is embedded, a query like *"top customers by revenue"* will match `v_sales_summary` in pgvector with a high similarity score. The retrieved DDL is then passed to the SQL generator, which (post-P1) also has the system prompt telling it to prefer this view. Both signals align.

**Thing 2 — Exclude `schema_metadata` and `business_glossary`**

After P0 creates these tables in the database, `get_table_names()` will pick them up on the next `/schemaJson` call. That causes two problems:

- The LLM enrichment step (`/schemaText`) wastes API calls generating "business purpose" docs for internal system tables
- They get embedded into pgvector and could appear as candidates for semantic search — queries like *"show descriptions"* might retrieve `schema_metadata` instead of a business table

**Fix:** Since `include_tables` in `SchemaExtractionConfig` already acts as a whitelist, `schema_metadata` and `business_glossary` are implicitly excluded from the table loop. The same whitelist is applied to the new view loop via `_table_allowed()`. No extra `exclude_tables` entry is needed as long as `v_sales_summary` is added to `include_tables`.

---

#### Files That Change

| File | Change |
|---|---|
| `schema_extractor/SchemaModels.py` | Add `object_type: str = "table"` and `view_definition: str \| None = None` to `TableSchema` dataclass |
| `schema_extractor/SchemaExtractor.py` | Add view extraction loop in `extract_database_schema()` + new `_extract_view_schema()` method |
| `server.py` | Add `"v_sales_summary"` to `include_tables` in the `/schemaJson` endpoint config |

---

#### Current Code vs After Fix

**`SchemaExtractor.py` — `extract_database_schema()` before:**

```python
def extract_database_schema(self) -> DatabaseSchema:
    tables = []
    for schema_name in self._get_schemas():
        if not self._schema_allowed(schema_name):
            continue
        for table_name in self.inspector.get_table_names(schema=schema_name):
            if not self._table_allowed(table_name):
                continue
            tables.append(self._extract_table_schema(schema_name, table_name))
            # views never extracted ❌
    return DatabaseSchema(database_name=..., tables=tables)
```

**After:**

```python
def extract_database_schema(self) -> DatabaseSchema:
    tables = []
    for schema_name in self._get_schemas():
        if not self._schema_allowed(schema_name):
            continue
        for table_name in self.inspector.get_table_names(schema=schema_name):
            if not self._table_allowed(table_name):
                continue
            tables.append(self._extract_table_schema(schema_name, table_name))
        # new: extract views (v_sales_summary)
        for view_name in self.inspector.get_view_names(schema=schema_name):
            if not self._table_allowed(view_name):
                continue
            tables.append(self._extract_view_schema(schema_name, view_name))
    return DatabaseSchema(database_name=..., tables=tables)

def _extract_view_schema(self, schema_name: str, view_name: str) -> TableSchema:
    columns = self._extract_columns(schema_name, view_name)
    view_definition = self.inspector.get_view_definition(view_name, schema=schema_name)
    return TableSchema(
        schema_name=schema_name,
        table_name=view_name,
        columns=columns,
        primary_keys=[],
        foreign_keys=[],
        object_type="view",
        view_definition=view_definition,
    )
```

**`server.py` — `/schemaJson` config before:**

```python
config = SchemaExtractionConfig(
    include_tables=["orders", "customer", "sales", "product", "inventory"]
)
```

**After:**

```python
config = SchemaExtractionConfig(
    include_tables=["orders", "customer", "sales", "product", "inventory", "v_sales_summary"]
)
```

---

#### After P2 Is Done — Re-run the Setup Pipeline

P2 only changes the extraction code. To make the view available for semantic search, the 3-step pipeline must be re-run once:

```
GET /schemaJson   → db_schema.json now includes v_sales_summary
GET /schemaText   → LLM generates business-context YAML for the view
GET /ingest       → v_sales_summary embedded into pgvector, retrievable by semantic search
```

From that point: a query like *"show revenue by customer"* → semantic search returns `v_sales_summary` → DDL injected into generation prompt → LLM generates `SELECT ... FROM v_sales_summary` instead of a 3-table JOIN.

---

### P1 — Deep Dive: Injecting `system_prompt.md` into the SQL Generator

#### The Problem in Plain English

The system makes two LLM calls per query. Only the first one receives `system_prompt.md`:

```
Router LLM (FAST model — DeepSeek-V4-Flash)     → gets system_prompt.md  ✅
SQL Generator LLM (SMART model — Llama-3.3-70B) → gets hardcoded generic prompt ❌
```

The router uses the system prompt to classify intent and identify relevant tables. But when actual SQL is generated — the most critical step — the LLM has no domain knowledge: no business definitions, no view preference, no SQL rules, no example patterns.

---

#### What the SQL Generator Currently Receives

`generate_sql.py:_create_generation_prompt()` (lines 103–138) sends only this to the SMART LLM:

```
You are an expert SQL query generator...

DATABASE SCHEMA:
-- Table: sales (Relevance: 0.91)
CREATE TABLE sales (...)      ← raw DDL from vector store

USER REQUEST:
show me top customers this year

REQUIREMENTS:
1. Generate ONLY SELECT queries
2. Use proper JOIN syntax
3. Include WHERE clauses
4. Use GROUP BY when needed
5. Limit results to max 10 rows   ← conflicts with system_prompt.md rule (should be 100)
6. Use PostgreSQL syntax
...
```

The LLM must infer joins, business term meanings, and query patterns from raw DDL alone.

---

#### What the SQL Generator Should Receive After the Fix

After prepending `system_prompt.md`, the SMART LLM receives full domain context:

```
# NL-to-SQL System Prompt — Sales Order Management & Inventory System

## ROLE
You are a PostgreSQL SQL query generator...

## PREFERRED VIEW
Use v_sales_summary for any query involving more than one table.

## FULL COLUMN REFERENCE
[all 5 tables with every column and type documented]

## v_sales_summary VIEW COLUMNS
sales_id, order_date, ..., customer_name, ..., product_name, ..., on_hand, ...

## BUSINESS DEFINITIONS
"revenue"       → SUM(sales.quantity * sales.unit_price)
"top customers" → GROUP BY customer_name ORDER BY SUM(quantity * unit_price) DESC
"this year"     → WHERE order_date >= date_trunc('year', CURRENT_DATE)
"pending orders"→ orders.order_status = 'PENDING'
...

## ORDER LIFECYCLE
PENDING → ORDERED → PROCESSING → PROCESSED → POSTED → DELIVERED → COMPLETED

## SQL GENERATION RULES
1. SELECT only — never INSERT/UPDATE/DELETE/DROP
2. Always use aliases (customer c, orders o, sales s, ...)
3. GROUP BY required for any aggregation
4. Default LIMIT 100
5. ILIKE for text search — never LIKE
6. Prefer v_sales_summary for multi-table queries
7. COALESCE for nulls in aggregations
8. COUNT DISTINCT for orders
...

## EXAMPLE QUERY PAIRS
Q: Top 10 customers by revenue this year
→ SELECT customer_name, SUM(quantity * unit_price) AS revenue
  FROM v_sales_summary
  WHERE order_date >= date_trunc('year', CURRENT_DATE)
  GROUP BY customer_name ORDER BY revenue DESC LIMIT 10;
[30+ more examples]

--- [dynamic section appended below] ---

## RETRIEVED SCHEMA CONTEXT
-- Table: v_sales_summary (Relevance: 0.94)
CREATE TABLE v_sales_summary (...)

## USER REQUEST
show me top customers this year
```

---

#### The Concrete Code Change

**File:** `src/ai_agentic_chatbot/agent/subgraphs/sql_query/nodes/generate_sql.py`

**Step 1 — Add import at the top of the file:**

```python
from ai_agentic_chatbot.utils.prompt_loader import get_system_prompt
```

**Step 2 — Update `_create_generation_prompt()` to prepend the system prompt:**

```python
# Before
def _create_generation_prompt(schema_text, user_query, previous_error, generation_attempts):
    base_prompt = f"""You are an expert SQL query generator...

DATABASE SCHEMA:
{schema_text}

USER REQUEST:
{user_query}

REQUIREMENTS:
...
Limit results to a reasonable number (max 10 rows)   ← wrong
..."""
    return base_prompt

# After
def _create_generation_prompt(schema_text, user_query, previous_error, generation_attempts):
    system_context = get_system_prompt()   # loads system_prompt.md at call time

    base_prompt = f"""{system_context}

---

## RETRIEVED SCHEMA CONTEXT
The following tables were retrieved as most relevant to the user's request:

{schema_text}

## USER REQUEST
{user_query}

Generate the SQL query following all rules and business definitions above.
Return ONLY the JSON object — no markdown, no explanation outside the JSON."""
    return base_prompt
```

No other files change. The retry logic (appending `previous_error` on subsequent attempts) is added after `base_prompt` exactly as it is today.

---

#### Impact by Query Type

| User Query | Without P1 | With P1 |
|---|---|---|
| "show revenue this month" | LLM guesses formula | Knows `revenue = SUM(quantity * unit_price)` from BUSINESS DEFINITIONS |
| "top customers this year" | Writes 3-table JOIN | Uses `v_sales_summary`, correct GROUP BY pattern from examples |
| "pending orders" | May use `'pending'` (wrong case) | Knows `order_status = 'PENDING'` from definitions |
| "find customer abc" | Uses `LIKE '%abc%'` | Uses `ILIKE '%abc%'` (Rule 6) |
| "sales this month" | May hardcode a date | Uses `date_trunc('month', CURRENT_DATE)` (Rule 12 + examples) |
| Row limit | Returns 10 rows | Returns 100 rows (Rule 5) |
| "gross profit" | Unknown — may error | Knows `SUM((sell_for - base_cost) * quantity)` |

---

#### Token Budget Consideration

`system_prompt.md` is ~930 lines (~5,000 tokens). Combined with retrieved schema DDL (~500 tokens), the total input per SQL generation call increases from ~800 tokens to ~5,800 tokens.

| | Current | After P1 |
|---|---|---|
| Input tokens per SQL call | ~800 | ~5,800 |
| Output tokens per SQL call | ~200 | ~200 (unchanged) |
| Context window used | <1% of 128k | ~4.5% of 128k |
| Latency impact | baseline | negligible — input tokens are fast |

Llama-3.3-70B is configured with `max_tokens: 20000` and a 128k context window. The larger input is well within limits and does not affect output speed meaningfully.

---

---

### P7 — Deep Dive: LLM Provider Resolution Bug — `provider` Argument Silently Ignored

> **GitHub Issue:** [#15](https://github.com/dccservicespl/ai-agentic-chatbot/issues/15)

#### Summary

Despite `config.yaml` having `default: azure_openai.fast` and call sites explicitly passing `LLMProvider.AZURE_OPENAI`, the app was actually running **Azure AI Foundry models** (DeepSeek-V4-Flash / Llama-3.3-70B) for all LLM calls. The `provider` argument to `get_llm()` was silently discarded and the `default` setting had no effect.

---

#### Three Bugs Working Together

**Bug 1 — `settings.py` strips the provider prefix from the default key (line 93–94)**

```python
default_model_key = llm_config.get("default", "azure_openai.fast")
if "." in default_model_key:
    default_model_key = default_model_key.split(".", 1)[1]   # "azure_openai.fast" → "fast"
```

The provider portion `"azure_openai"` is thrown away immediately. Only `"fast"` survives.

**Bug 2 — Both providers stored under the same short key — last one parsed wins**

`_parse_config()` builds a flat `models` dict keyed by `"fast"`, `"smart"`, `"embedding"`. Both `azure_openai` and `azure_ai_foundry` define `fast` and `smart`. Since `azure_ai_foundry` appears second in `config.yaml`, it silently overwrites the azure_openai entries:

```
models["fast"]  = AZURE_OPENAI (gpt-4o-mini)      ← stored first
models["smart"] = AZURE_OPENAI (gpt-4.1)           ← stored first
models["fast"]  = AZURE_AI_FOUNDRY (DeepSeek)      ← overwrites ❌
models["smart"] = AZURE_AI_FOUNDRY (Llama-3.3-70B) ← overwrites ❌
```

**Bug 3 — `factory.py` ignores the `provider` argument entirely (lines 56–71)**

```python
def get_llm(self, provider=None, model=None):
    ...
    model_key = model.value   # just "fast" or "smart" — provider never used
    model_config = self._settings.get_model_config(model_key)
    # provider argument is discarded — get_llm(AZURE_OPENAI, SMART)
    # and get_llm(AZURE_AI_FOUNDRY, SMART) return identical results
```

---

#### Actual Models Used Before Fix

| Call site | Code says | Key resolved | Model actually used |
|---|---|---|---|
| `router.py:62` | `get_llm()` | `"fast"` → AZURE_AI_FOUNDRY | **DeepSeek-V4-Flash** ❌ |
| `graph.py:13` | `get_llm(AZURE_OPENAI, FAST)` | `"fast"` → AZURE_AI_FOUNDRY | **DeepSeek-V4-Flash** ❌ |
| `generate_sql.py:50` | `get_llm(AZURE_OPENAI, SMART)` | `"smart"` → AZURE_AI_FOUNDRY | **Llama-3.3-70B** ❌ |
| `embedding` | `get_embedding()` | `"embedding"` → AZURE_OPENAI | **text-embedding-3-small** ✅ |

---

#### The Fix — Full Provider-Qualified Keys Throughout

The `config.yaml` design (`default: azure_openai.fast`, separate provider blocks) is correct. The fix is to stop stripping the provider prefix and use full keys `"azure_openai.fast"`, `"azure_openai.smart"`, `"azure_ai_foundry.fast"` etc. everywhere so both providers coexist without collision.

**`settings.py` changes:**

1. Keep the full default key — do not strip the provider prefix
2. Store models under full keys `"{provider}.{model_key}"` instead of just `"{model_key}"`

```python
# Before
default_model_key = default.split(".", 1)[1]          # "fast"
models["fast"] = ModelConfiguration(AZURE_OPENAI, ...) # overwritten by foundry

# After
default_model_key = default                            # "azure_openai.fast"
models["azure_openai.fast"]      = ModelConfiguration(AZURE_OPENAI, FAST, ...)
models["azure_openai.smart"]     = ModelConfiguration(AZURE_OPENAI, SMART, ...)
models["azure_ai_foundry.fast"]  = ModelConfiguration(AZURE_AI_FOUNDRY, FAST, ...)
models["azure_ai_foundry.smart"] = ModelConfiguration(AZURE_AI_FOUNDRY, SMART, ...)
```

**`factory.py` changes:**

Build the lookup key from provider + model when both are supplied:

```python
# Before — provider ignored
model_key = model.value   # "smart"

# After — uses provider to build full key
if provider is not None and model is not None:
    model_key = f"{provider.value}.{model.value}"   # "azure_openai.smart"
elif model is not None:
    model_key = model.value                          # short key fallback (shouldn't occur)
else:
    model_key = self._settings.default_model        # "azure_openai.fast" from config
```

---

#### Models After Fix

| Call | Key resolved | Model used |
|---|---|---|
| `get_llm()` | `"azure_openai.fast"` (default) | gpt-4o-mini ✅ |
| `get_llm(AZURE_OPENAI, FAST)` | `"azure_openai.fast"` | gpt-4o-mini ✅ |
| `get_llm(AZURE_OPENAI, SMART)` | `"azure_openai.smart"` | gpt-4.1 ✅ |
| `get_llm(AZURE_AI_FOUNDRY, FAST)` | `"azure_ai_foundry.fast"` | DeepSeek-V4-Flash ✅ |
| `get_llm(AZURE_AI_FOUNDRY, SMART)` | `"azure_ai_foundry.smart"` | Llama-3.3-70B ✅ |

Both providers coexist. Each call site gets exactly the model it requests. Switching a call site from one provider to another is a one-argument change.

---

#### Files Changed

| File | Change |
|---|---|
| `infrastructure/llm/settings.py` | Keep full key as default; store models under `"{provider}.{model_key}"` |
| `infrastructure/llm/factory.py` | Build lookup key as `"{provider.value}.{model.value}"` when provider is supplied |

---

### P8 — Deep Dive: `/ingest` Duplicates Schema Rows on Every Run

#### Summary

Every call to `GET /ingest` **always inserts new rows** into the pgvector table — it never updates or replaces existing ones. After N ingest runs with 5 tables, the collection holds N×5 rows. Semantic search returns duplicate entries for the same table, consuming the top-k budget and crowding out other relevant tables.

---

#### Root Cause — Three Missing Pieces

**1. No stable document IDs in `VectorSchemaBuilder`**

`vector_schema_builder.py:build_all_tables()` returns chunks with no `id` field:

```python
{
    "table_name": "sales",
    "content": "...",
    "metadata": { "table_name": "sales", ... }
    # no "id" field ❌
}
```

Without an ID, LangChain generates a **random UUID** per document on every call. Same table ingested 5 times → 5 different UUIDs → 5 separate rows.

**2. `add_documents()` called without `ids=` — upsert never fires**

`pgvector_store.py:47`:

```python
self._vectorstore.add_documents(documents)   # no ids= argument ❌
```

LangChain PGVector supports upsert when `ids=` is passed — it executes `INSERT ... ON CONFLICT (id) DO UPDATE`. Without IDs the conflict check never fires and every call is a pure `INSERT`.

**3. No pre-ingest cleanup**

There is no `delete_collection()`, filter-based delete, or any cleanup before inserting. Old rows accumulate indefinitely.

---

#### Impact on Search Quality

After multiple ingest runs, `search()` returns duplicate `table_name` values:

```
[("sales", 0.91), ("sales", 0.90), ("sales", 0.89), ("customer", 0.88), ("customer", 0.87)]
```

The top-k limit (`k=5`) is consumed by duplicates of the same table, crowding out other relevant tables. Every downstream step — router hint boost, `_expand_related_tables()`, SQL generation — operates on this polluted result set.

---

#### Fix 1 — Stable Deterministic IDs in `VectorSchemaBuilder`

**File:** `schema_extractor/vector_schema_builder.py`

Use `uuid.uuid5(uuid.NAMESPACE_DNS, table_name)` — produces the **same UUID for the same table name on every run**. Add it to every chunk in `build_all_tables()`:

```python
import uuid

chunk["id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, table["table_name"]))
```

This is the prerequisite for Fix 2 — without stable IDs, upsert has nothing to key on.

---

#### Fix 2 — Pass `ids=` to `add_documents()` to Enable Upsert

**File:** `infrastructure/vector_store/pgvector_store.py:ingest()`

Extract IDs from chunks and pass them explicitly. PGVector then executes `INSERT ... ON CONFLICT (id) DO UPDATE SET embedding=..., document=..., cmetadata=...`:

```python
def ingest(self, table_chunks: List[Dict]) -> None:
    documents, ids = [], []
    for chunk in table_chunks:
        documents.append(Document(page_content=chunk["content"], metadata=chunk["metadata"]))
        ids.append(chunk["id"])
    self._vectorstore.add_documents(documents, ids=ids)
```

---

#### Fix 3 — `reset_collection()` to Clean Up Existing Duplicates

**File:** `infrastructure/vector_store/pgvector_store.py`

If `/ingest` was called multiple times before this fix, the collection already has duplicates. Add a `reset_collection()` method that drops and recreates the collection, and expose a `force_reset` flag on the `/ingest` endpoint for one-time cleanup:

```python
def reset_collection(self) -> None:
    self._vectorstore.delete_collection()
    self._vectorstore.create_collection()
```

---

#### Fix 4 — Deduplicate by `table_name` in `search()` (Safety Net)

**File:** `infrastructure/vector_store/pgvector_store.py:search()`

Even with upsert working correctly, add a deduplication step that keeps only the highest-scoring entry per `table_name`. Guards against any future accidental duplicates:

```python
seen: Dict[str, float] = {}
for table_name, score in all_results:
    if table_name not in seen or score > seen[table_name]:
        seen[table_name] = score
return sorted(seen.items(), key=lambda x: x[1], reverse=True)
```

---

#### TODO Summary

| Fix | File | What it does | Priority |
|---|---|---|---|
| 1 — Stable IDs | `vector_schema_builder.py` | `uuid.uuid5` per table — same ID every run | P0 — prerequisite |
| 2 — Upsert via IDs | `pgvector_store.py:ingest()` | `add_documents(docs, ids=ids)` → INSERT OR UPDATE | P0 — stops new duplicates |
| 3 — Reset existing duplicates | `pgvector_store.py` | `reset_collection()` + `force_reset` flag on `/ingest` | P1 — one-time cleanup |
| 4 — Search deduplication | `pgvector_store.py:search()` | Keep highest score per `table_name` | P1 — safety net |

---

### P3 — Deep Dive: Router Schema Summary `.items()` Bug

#### Summary

The router LLM receives a garbled list of "available tables" — it sees `database_name`, `version`, `tables` as table names instead of the actual business tables (`customer`, `orders`, `sales`, etc.). This breaks intent classification, the answerability check, and the `relevant_tables` hint passed to the SQL subgraph.

---

#### The Full Data Flow

```
GET /schemaText
  └── generate_schema_summary()
        └── writes schema_summary.json  ($SCHEMA_SUMMARY_PATH)

router_node (every user query)
  └── schema_loader.load_schema_summary()  → returns full JSON dict
  └── formats dict into text → injected into router LLM prompt
```

---

#### What `schema_summary.json` Actually Contains

`generate_schema_summary()` in `transform_schema_to_text.py:86–103` writes:

```json
{
    "database_name": "ai_chatbot_db",
    "version": "v1",
    "tables": [
        { "table": "customer",  "bussiness_purpose": "Master list of all customers...", "example_questions": [...] },
        { "table": "orders",    "bussiness_purpose": "Customer orders with status tracking...", "example_questions": [...] },
        { "table": "sales",     "bussiness_purpose": "Individual line items of each order...", "example_questions": [...] },
        { "table": "product",   "bussiness_purpose": "Product master catalog...", "example_questions": [...] },
        { "table": "inventory", "bussiness_purpose": "Current stock levels per product...", "example_questions": [...] }
    ]
}
```

The actual table data is nested inside the `"tables"` key.

---

#### The Bug — `router.py:74-78`

```python
schema_summary = schema_loader.load_schema_summary()   # returns full dict above

schema_summary_text = "\n".join(
    [f"- {table}: {desc}" for table, desc in schema_summary.items()]
)
```

`.items()` on the top-level dict yields the top-level keys only:

```
("database_name", "ai_chatbot_db")
("version",       "v1")
("tables",        [{"table": "customer", ...}, ...])   ← entire list as one value
```

So `schema_summary_text` injected into the router prompt becomes:

```
- database_name: ai_chatbot_db
- version: v1
- tables: [{'table': 'customer', 'bussiness_purpose': '...'}, ...]
```

The router LLM sees `database_name`, `version`, and `tables` as the available "table names".

---

#### Second Bug — `router.py:98` (Same Block)

```python
response_msg = f"\n\nI can help you with: {', '.join(schema_summary.keys())}."
```

`schema_summary.keys()` returns `["database_name", "version", "tables"]` — again the top-level keys, not table names. When the router tells the user what data it can help with, it says *"I can help you with: database_name, version, tables"*.

---

#### Impact on Router Behaviour

| Failure | Cause |
|---|---|
| `is_answerable: False` for valid queries | LLM sees no `customer`/`orders` table — concludes data is unavailable |
| `relevant_tables` hint is wrong | LLM returns `["database_name"]` instead of `["customer", "sales"]` — hurts pgvector boost |
| Out-of-scope guard may break | LLM can't parse garbled list, may mark everything as answerable |
| Wrong "I can help with" message | Uses top-level JSON keys instead of table names |

---

#### The Fix

**File:** `agent/router.py`

**Fix 1 — lines 74–78: replace `.items()` with nested table loop**

```python
# Before
schema_summary_text = "\n".join(
    [f"- {table}: {desc}" for table, desc in schema_summary.items()]
)

# After
schema_summary_text = "\n".join(
    [
        f"- {t['table']}: {t['bussiness_purpose']}"
        for t in schema_summary.get("tables", [])
    ]
)
```

Router prompt now receives:

```
- customer: Master list of all customers. Each customer has a unique code used across orders and sales.
- orders: Customer orders with status tracking from PENDING through COMPLETED or VOID.
- sales: Individual line items of each order. Each row is one product sold in one order.
- product: Product master catalog with pricing, status, and category information.
- inventory: Current stock levels per product including on-hand, available, and expected quantities.
```

**Fix 2 — line 98: replace `.keys()` with actual table names**

```python
# Before
response_msg = f"\n\nI can help you with: {', '.join(schema_summary.keys())}."

# After
table_names = [t["table"] for t in schema_summary.get("tables", [])]
response_msg = f"\n\nI can help you with: {', '.join(table_names)}."
```

---

#### Files Changed

| File | Lines | Change |
|---|---|---|
| `agent/router.py` | 74–78 | Replace `.items()` with `.get("tables", [])` loop using `t["table"]` and `t["bussiness_purpose"]` |
| `agent/router.py` | 98 | Replace `.keys()` with list of `t["table"]` from nested tables |

*Generated: 2026-05-29 | Updated: 2026-06-10 | Repository: `ai-agentic-chatbot` | Branch: `develop`*
