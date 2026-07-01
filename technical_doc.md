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
22. [Visualization State Leak on Non-Contextual Queries — Bug Fix Plan](#22-visualization-state-leak-on-non-contextual-queries--bug-fix-plan)
23. [ROUND() on `double precision` Crashes Revenue-Share Pie Chart Queries](#23-round-on-double-precision-crashes-revenue-share-pie-chart-queries)
24. [`/stream` Doesn't Surface Intermediate Progress — Analysis & Plan](#24-stream-doesnt-surface-intermediate-progress--analysis--plan)
25. [Visualization Analysis — Feature Plan](#25-visualization-analysis--feature-plan)
26. [User Authentication Module — Implementation Plan](#26-user-authentication-module--implementation-plan)
27. [JWT Refresh Token System — Design & Implementation Plan](#27-jwt-refresh-token-system--design--implementation-plan)
28. [Alembic vs Docker — Migration System Clash Analysis](#28-alembic-vs-docker--migration-system-clash-analysis)
29. [User Prompt Management — Implementation Plan ✅ COMPLETED](#29-user-prompt-management--implementation-plan-completed)
30. [Multi-Database Context Support — Architecture & TODO List](#30-multi-database-context-support--architecture--todo-list)
31. [Database Structure for Multi-Context — Decision & Schema Design](#31-database-structure-for-multi-context--decision--schema-design)

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

**File:** `src/ai_agentic_chatbot/agent/nodes/visualizer.py`

### Overview

The Visualization Engine is a **LangGraph node** in the agent pipeline. Its sole responsibility is to receive SQL query results and automatically decide the best chart or display type — KPI card, line chart, bar chart, pie chart, or table. The decision is entirely rule-based (heuristic), with a TODO to upgrade to LLM-assisted selection in a future iteration.

---

### Structure — `VisualizationNode` Class

#### `determine_visualization(state)` — lines 16–53

The **public entry point** called by LangGraph. It reads three keys from the agent `state` dict:

| Key | Source |
|---|---|
| `query_result` | Rows returned from SQL execution node |
| `generated_sql` | The SQL string that was executed |
| `explanation` | LLM-generated plain-English explanation |

**Short-circuit:** if `query_result` is empty, returns a `"text"` type payload immediately with a "No Results" message — no DataFrame analysis occurs.

Otherwise, converts results to a Pandas DataFrame and delegates to `_apply_heuristics()`.

> **Note:** Line 55 has a redundant module-level `logger = get_logger(__name__)` statement declared inside the class body but outside `__init__`. It runs at class definition time and is harmless but redundant — `logger` is already declared at module level on line 7.

---

#### `_apply_heuristics(df, sql_query, explanation)` — lines 57–174

The **decision engine**. Applies a priority-ordered waterfall of rules against the DataFrame's shape, column types, and column names. The first rule that matches wins.

| Priority | Condition | Visualization Type |
|---|---|---|
| 1 | 1 row × 1 column | `kpi` — single metric card |
| 2 | 2 cols, first col parseable as date/time | `line_chart` — time series |
| 3 | 2 cols, ≤20 rows, string + numeric columns | `bar_chart` — categorical comparison |
| 4 | 2 cols, ≤8 rows, second col name contains "percent / share / proportion" | `pie_chart` — distribution |
| 5 | ≥3 cols, ≤50 rows | `table` — detailed multi-column view |
| 6 (fallback) | Everything else | `table` — capped at 100 rows, paginated |

**Priority conflict note:** Rules 2 and 3 both check `num_cols == 2`. Since rule 2 is evaluated first, any 2-column dataset where the first column is a date will always produce a `line_chart` — even if the row count is ≤20. This is intentional: time series takes priority over categorical bar charts.

---

### Helper Methods

| Method | File Location | Purpose |
|---|---|---|
| `_is_date_column(series)` | line 178 | Guards against numeric series first (`is_numeric_dtype` → `False`), then samples the first 5 values and attempts `pd.to_datetime()`. Returns `True` only if the series is non-numeric and all sampled values parse without error. |
| `_format_kpi_value(value, column_name)` | line 186 | Formats a raw number for display: `$1,234.56` for money columns, `12.3%` for ratios, `1,234` for counts. Driven by keyword matching on the column name. |
| `_detect_value_format(value, column_name)` | line 224 | Returns a format type string (`"currency"`, `"percentage"`, `"integer"`, `"decimal"`, `"text"`) consumed by the frontend for CSS/display styling. |
| `_beautify_column_name(column_name)` | line 254 | Converts snake_case or kebab-case column names to Title Case: `total_sales_amount` → `Total Sales Amount`. |
| `_create_payload(...)` | line 258 | Builds the standardized output dict returned by every branch: `type`, `title`, `data`, `columns`, `config`, `summary`, `row_count`. |

---

### Value Formatting Rules (`_format_kpi_value` / `_detect_value_format`)

Both methods apply the same keyword-matching logic on the column name:

| Column Name Keywords | Display Format | Format Type String |
|---|---|---|
| `sales`, `revenue`, `amount`, `price`, `cost`, `total`, `value` | `$1,234.56` | `"currency"` |
| `percent`, `rate`, `ratio` (value 0–1) | `12.5%` | `"percentage"` |
| `percent`, `rate`, `ratio` (value > 1) | `12.5%` (no conversion) | `"percentage"` |
| `count`, `number`, `qty`, `quantity` | `1,234` (integer) | `"integer"` |
| Other numeric ≥ 1000 | `1,234.00` | `"decimal"` |
| Other numeric < 1000 | `0.42` | `"decimal"` |
| Non-numeric | raw string | `"text"` |

---

### LangGraph Entry Point — `visualizer_node(state)` — lines 278–281

A module-level function that LangGraph registers as a graph node. Instantiates `VisualizationNode` on each call and delegates to `determine_visualization()`.

```python
def visualizer_node(state: dict) -> dict:
    visualizer = VisualizationNode()
    return visualizer.determine_visualization(state)
```

---

### Data Flow

```
LangGraph agent state
    { query_result, generated_sql, explanation }
         │
         ▼
visualizer_node(state)
         │
         ▼
VisualizationNode.determine_visualization()
         │
         ├── empty results? → return { type: "text", ... }
         │
         ▼
pd.DataFrame(query_result)  →  shape analysis
         │
         ▼
_apply_heuristics()  —  priority waterfall
         │
         ▼
_create_payload()  →  standardized dict
         │
         ▼
state["visualization"] = {
    type, title, data, columns, config, summary, row_count
}
```

---

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
    "x_label": "Column A",
    "y_label": "Column B"
  },
  "summary": "One-sentence insight about the data.",
  "row_count": 10
}
```

**`config` shape by chart type:**

| Chart Type | Config Keys |
|---|---|
| `kpi` | `value` (formatted string), `metric` (column name), `format` (type string) |
| `line_chart` | `x_axis`, `y_axis`, `x_label`, `y_label` |
| `bar_chart` | `x_axis`, `y_axis`, `x_label`, `y_label` |
| `pie_chart` | `category`, `value`, `category_label`, `value_label` |
| `table` | `columns` (list), `highlight_numeric` (bool), `sortable` (bool), `total_rows` (int), `paginated` (bool) |

---

### Known Issues / TODOs

| # | Location | Issue |
|---|---|---|
| 1 | line 48 | `# TODO: apply intelligent heuristics using LLMs` — current logic is purely rule-based. Plan is to let an LLM choose the chart type based on data shape + SQL context. |
| 2 | line 65 | `logger.info("df", df.head())` is a **bug**: `logger.info()` does not accept two positional args like `print()`. This either silently logs the string `"df"` and discards the DataFrame, or raises a `TypeError` depending on the logging handler. Should be `logger.info("df head: %s", df.head())` or `logger.debug(df.head().to_string())`. |
| 3 | line 55 | Duplicate `logger = get_logger(__name__)` at class-body scope — already declared at module level on line 7. Harmless but should be removed. |

---

### Date Formatting — `dd-mm-yyyy`

**Problem:** PostgreSQL returns date/datetime values as Python `datetime` objects. `_serialize_value()` in `execute_query.py` converts them via `.isoformat()`, producing strings like `2026-01-01T00:00:00+00:00`. These raw ISO strings were passed through to the visualization payload unchanged.

**Fix location:** `visualizer.py` — formatting is applied inside the visualization pipeline only. The raw `query_result` in the agent state still holds the ISO string, so other consumers (raw API callers, retry logic) are unaffected.

#### New method — `_format_date_columns(df, date_flags)` (line 188)

```python
def _format_date_columns(self, df: pd.DataFrame, date_flags: list) -> pd.DataFrame:
    """Reformat detected date columns to dd-mm-yyyy, stripping time and timezone."""
    df = df.copy()
    for i, is_date in enumerate(date_flags):
        if is_date:
            col = df.columns[i]
            df[col] = (
                pd.to_datetime(df[col], utc=True, errors="coerce")
                .dt.strftime("%d-%m-%Y")
            )
    return df
```

- `utc=True` — normalises timezone-aware ISO strings (e.g. `+00:00`, `+05:30`) before formatting.
- `errors="coerce"` — unparseable values become `NaT` → `None` instead of raising.
- Time component is always stripped — `2026-01-15T14:35:22+00:00` → `15-01-2026`.

#### Change in `_apply_heuristics()` (lines 65–67 & 90)

```python
# Before (detect and format happened independently, out of order)
if self._is_date_column(df.iloc[:, 0]):   # line chart check on raw ISO string
    ...

# After (detect first on raw data, then format, then use pre-computed flags)
date_flags = [self._is_date_column(df.iloc[:, i]) for i in range(num_cols)]
df = self._format_date_columns(df, date_flags)

# line chart branch now uses the pre-computed flag
if date_flags[0]:
    ...
```

**Why detect before format:** After reformatting to `dd-mm-yyyy`, the string `15-01-2026` is ambiguous to `pd.to_datetime()` — pandas may parse it as `YYYY-DD-MM`, causing `_is_date_column()` to return `False` and misclassify a time-series result as a bar chart. Detecting on the original ISO strings ensures correct chart type selection.

#### Result

| Before | After |
|---|---|
| `"2026-01-01T00:00:00+00:00"` | `"01-01-2026"` |
| `"2026-01-15T14:35:22+05:30"` | `"15-01-2026"` |

Applies to all visualization types — table cells, KPI values, and line chart X-axis labels.

---

### Bug Fix — Integer Columns Misidentified as Dates (`_is_date_column`)

**Symptom:** Queries returning integer aggregates such as `COUNT(DISTINCT order_no) AS order_count` showed values like `01-01-1970` instead of the actual count, and the visualization type was `table` instead of `bar_chart`.

**Root cause — 3-bug cascade:**

`pd.to_datetime()` silently accepts plain integers, treating them as **nanoseconds since Unix epoch**. A `COUNT` result like `42` parsed without error:

```
pd.to_datetime(42) → Timestamp('1970-01-01 00:00:00.000000042')
```

This caused a cascade across three functions:

| Step | Bug | Effect |
|---|---|---|
| 1 | `_is_date_column([42, 38, ...])` → `True` (should be `False`) | Integer column wrongly flagged as a date |
| 2 | `_format_date_columns()` converts `42` → `"01-01-1970"` | COUNT values replaced with epoch date strings |
| 3 | Bar chart check: `is_numeric_dtype("01-01-1970")` → `False` | Bar chart skipped, falls through to table |

```
order_count = [42, 38, 35, ...]   ← integers from COUNT()
      │
      ▼  _is_date_column() — pd.to_datetime(42) succeeds → True   ← BUG 1
      │
      ▼  _format_date_columns() — 42 → "01-01-1970"               ← BUG 2
      │
      ▼  is_numeric_dtype("01-01-1970") → False → bar chart skipped ← BUG 3
      │
      ▼  type: "table"   (wrong — should be bar_chart)
```

**Fix — one guard line added to `_is_date_column()` (line 179):**

```python
def _is_date_column(self, series) -> bool:
    if pd.api.types.is_numeric_dtype(series):   # ← added guard
        return False
    try:
        sample_size = min(5, len(series))
        sample = series.head(sample_size)
        pd.to_datetime(sample, errors="raise")
        return True
    except (ValueError, TypeError):
        return False
```

`is_numeric_dtype` returns `True` for both `int` and `float` dtypes, so the guard covers COUNT, SUM, AVG, and any other numeric aggregate. All three cascade bugs resolve from this single change.

---

### Visualization Type Prediction — Improvement Plan

**GitHub Issue:** [#17](https://github.com/dccservicespl/ai-agentic-chatbot/issues/17)

The current heuristic is a **shape-only waterfall** — it looks exclusively at row count, column count, and column data types. Four richer signal sources already available in the pipeline are currently unused:

| Signal Source | Available | Currently Used |
|---|---|---|
| SQL query structure (`sql_lower`) | Yes — computed at line 63 | No — dead variable |
| LLM explanation (`explanation`) | Yes — passed into the method | No — never read |
| Data values (cardinality, sum, distribution) | Yes — DataFrame is present | Partial — dtype only |
| User's original NL question | Yes — in agent state | No — not passed to visualizer |

#### Known problems in the current heuristic

| # | Problem | Impact |
|---|---|---|
| 1 | Pie chart rule runs **after** bar chart rule — both require 2 cols + numeric second col, so bar chart always wins first | Pie chart is structurally unreachable for real data |
| 2 | `sql_lower` is computed but never read | Temporal grouping, percentage calculation, ranking patterns are ignored |
| 3 | No data-value analysis | Values summing to ≈100, low cardinality, and distribution shape are never checked |
| 4 | No NL intent analysis | `explanation` field arrives in the method but is never read |
| 5 | Bar chart is the only real 2-column selector | Ranking, comparison, status breakdown, and distribution all produce bar chart — no distinction |

---

#### P1 — Fix structural ordering + expand pie chart triggers

**Status:** ✅ Complete

**What:** Move the pie chart rule before the bar chart rule. Expand pie chart detection beyond column-name keyword matching by adding two new signals — SQL percentage keywords and data-value sum analysis.

**Why:** The current rule order makes pie chart unreachable. Bar chart fires first for any 2-column string + numeric result, regardless of whether the data represents a distribution or a ranking.

**Final rule order after P1:**

```
KPI       → 1 row × 1 col
Line      → 2 cols, first col is date
Pie       → 2 cols, ≤ 8 rows, (keyword OR sum ≈ 100 OR SQL signal)
Bar       → 2 cols, ≤ 20 rows, string + numeric
Table     → 3+ cols or fallback
```

##### P1 — TODO List

---

**TODO 1 — Add `sql_has_percentage` signal in `_apply_heuristics()`** ✅

Location: `_apply_heuristics()` — after `sql_lower = sql_query.lower()` (line 63)

Compute a boolean from the SQL text that flags whether the query is calculating a percentage. Keywords detected: `"100.0"`, `"* 100"`, `"*100"`, `"/ sum"`, `"/sum"`, `"percent"`.

---

**TODO 2 — Add `_is_percentage_data(series)` helper method** ✅

Location: new method, added after `_is_date_column()`

Logic: if the numeric values in the series sum to between `99.0` and `101.0`, the column is a percentage distribution. Numeric type guard added — non-numeric series returns `False` immediately. `dropna()` applied before sum to handle NULLs from the DB.

> **Edge case noted:** COUNT values that coincidentally sum to 100 (e.g., a table with exactly 100 rows split into 2 categories: 80 / 20) will also trigger this check. Rare in practice and the behaviour is acceptable, but worth knowing.

---

**TODO 3 — Move pie chart block BEFORE bar chart block** ✅

Location: lines 104–145 (original)

Swapped the position of the two blocks — no logic inside either block changed, only the order:

```
Before:  line → bar → pie → table
After:   line → pie → bar → table
```

---

**TODO 4 — Expand pie chart trigger conditions** ✅

Location: inside the pie chart block

Replaced the single column-name keyword check with three OR-connected conditions:

| # | Condition | Signal type |
|---|---|---|
| A | Column name contains `percent` / `percentage` / `share` / `proportion` | Existing — kept |
| B | `_is_percentage_data(second_col)` returns True | New — values sum to ≈ 100 |
| C | `sql_has_percentage` is True | New — SQL has percentage calculation |

All three still require `num_cols == 2` and `num_rows <= 8`. Row limit unchanged — more than 8 pie slices is unreadable regardless.

---

**TODO 5 — Verify bar chart is unaffected** ✅

Traced through 5 real-world scenarios — no adjustment to the bar chart block was needed:

| Scenario | rows | Pie fires? | Result |
|---|---|---|---|
| Top 10 customers by revenue | 10 | No (> 8 rows) | Bar chart ✓ |
| Revenue share by brand (sum ≈ 100) | 5 | Yes (`is_percentage_named`) | Pie chart ✓ |
| Activation status % | 2 | Yes (`is_percentage_named`) | Pie chart ✓ |
| Order count by status (sum = 320) | 5 | No (all signals False) | Bar chart ✓ |
| Monthly revenue (date col) | N | Line fires first | Line chart ✓ |

---

**Execution order completed**

```
TODO 1 ✅ → TODO 2 ✅ → TODO 3 ✅ → TODO 4 ✅ → TODO 5 ✅
(signal)     (helper)    (reorder)   (expand)    (verify)
```

---

#### P2 — Use NL intent from the explanation field

**Status:** Pending — to be planned after P1 is complete

**What:** The LLM-generated `explanation` already arrives in `_apply_heuristics()` as a parameter but is never read. Extract intent keywords from it to guide chart selection before shape analysis runs.

**Intent keyword map (draft):**

| Keywords in explanation | Chart hint |
|---|---|
| "trend", "over time", "monthly", "weekly", "daily" | Line chart |
| "top N", "best", "worst", "rank", "highest", "lowest" | Bar chart |
| "breakdown", "distribution", "share", "portion", "split" | Pie chart |
| "compare", "vs", "versus", "compared to" | Bar chart |
| "list", "all", "details", "full", "show" | Table |

---

#### P3 — Use SQL structural patterns

**Status:** Pending — to be planned after P1 is complete

**What:** `sql_lower` is computed at line 63 but never read beyond P1's percentage signal. Extend its use to detect broader structural patterns in the SQL.

**Pattern map (draft):**

| SQL Pattern | Chart hint |
|---|---|
| `date_trunc`, `interval`, `extract` | Line chart (temporal grouping) |
| `order by ... desc limit` | Bar chart (ranking) |
| `100.0`, `/ sum`, `* 100` | Pie chart (percentage) |
| `select count(*) from ... (no group by)` | KPI |
| Multiple aggregates in SELECT | Table |

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

---

## 22. 🐛 Visualization State Leak on Non-Contextual Queries — Bug Fix Plan

**Files:** `agent/state.py`, `agent/router.py`, `agent/graph.py`, `server.py`, `prompts/router_prompts.md`

### Observed Behaviour

After running a SQL query that produced a chart, asking an unrelated conversational question in the **same `thread_id`** (e.g. `"how are you"`) returns the previous chart again instead of `null`:

```json
{
  "content": "I'm just a program, but I'm here and ready to help you! How can I assist you today?",
  "visualization": {
    "type": "pie_chart",
    "title": "Distribution of Delivery Method",
    "data": [
      {"delivery_method": "DELIVERY", "percentage": 94.2},
      {"delivery_method": "PICKUP", "percentage": 5.8}
    ],
    "columns": ["delivery_method", "percentage"],
    "config": {"category": "delivery_method", "value": "percentage", "category_label": "Delivery Method", "value_label": "Percentage"},
    "summary": "Distribution across 2 categories.",
    "row_count": 2
  }
}
```

This `pie_chart` is leftover from a prior turn's SQL query — `"how are you"` never touched the database.

---

### Root Cause — Two Compounding Bugs + One Contributing Factor

**Bug 1 — `visualization` has no reducer and the graph is checkpointed per `thread_id`**

`AgentState.visualization` (`agent/state.py:9`) is a plain optional field with no `Annotated` reducer. `build_graph()` compiles with `MemorySaver` (`agent/graph.py:416`), so the **full state persists across turns** within a `thread_id`. `greeting_node`, `fallback_node`, and `clarification_node` (`agent/graph.py:20-37`) never return a `visualization` key — and in LangGraph, a key that a node doesn't return is left untouched in the checkpoint, not cleared. So whatever the *last* `sql_query_node` wrote stays in state indefinitely until another SQL query overwrites it.

**Bug 2 — `server.py` republishes the last known visualization unconditionally**

```python
# server.py:260-265
response_data = {
    "content": content,
    "visualization": accumulated_state.get("visualization"),  # always attached, regardless of intent
}
```

Every SSE chunk attaches `accumulated_state.get("visualization")` no matter what the current turn's intent was.

**Contributing factor — router schema/prompt mismatch**

`router_prompts.md:12` describes a third intent, `out_of_scope`, but `RouterDecision.intent` (`agent/router.py:32`) only allows `Literal["greeting", "sql_query", "nonsense"]`. There is no enum value matching the prompt's instructions, so casual chit-chat with no data intent (`"how are you"`, `"tell me a joke"`) gets folded into `greeting` by the structured-output LLM, which is why it received a chatty LLM-generated reply rather than a scoped one.

---

### Impact

| Dimension | Effect |
|---|---|
| **Correctness** | Stale chart from an earlier, unrelated query is shown alongside an unrelated text reply — misleading to the user |
| **Trust** | Users may believe the new message ("how are you") legitimately produced that data, or that the system is malfunctioning |
| **Cost** | Every non-SQL turn (including pure chit-chat) still invokes `fast_llm` via `greeting_node` or `fallback_node`, even when a static reply would suffice |

---

### Decision — Greeting vs. Chit-Chat Behaviour

Confirmed with stakeholder: real greetings (`"hi"`, `"thanks"`) keep their LLM-generated warm reply via `greeting_node`. Only genuine chit-chat/off-topic/nonsense input (`"how are you"`, `"tell me a joke"`, gibberish) should get a **static, no-LLM-call** redirect message — these never need a generative answer.

---

### Fix / TODO List

- [x] **`agent/router.py` — `RouterNode.classify()`:** every return branch includes `"visualization": None, "relevant_tables": None` so each new turn starts clean. `sql_query_node` runs after the router within the same turn, so it still overwrites these with real values when the turn is a SQL query.
- [x] **`server.py:260-265`:** only attach `visualization` to the SSE payload when the turn's final state indicates a SQL turn, instead of unconditionally echoing `accumulated_state.get("visualization")`.
- [x] **`agent/router.py` (`RouterDecision.intent`) + `prompts/router_prompts.md`:** add a real `out_of_scope` value distinct from `greeting`, and align the prompt's category descriptions with the actual enum values.
- [x] **`agent/graph.py`:** wire `out_of_scope` (and `nonsense`) to a static, no-LLM path. `greeting_node` keeps calling `fast_llm` unchanged for real greetings.
- [x] **Regression test** (same `thread_id`): chart-producing SQL query → `"how are you"` (static reply, `visualization: null`, no LLM call) → `"hi"` (LLM-generated greeting, `visualization: null`) → gibberish (static fallback, `visualization: null`).
- [x] **`agent/router.py`:** defensively reset `next_step` in the same way as `visualization`/`relevant_tables`, for the same class of risk if the graph grows additional branches later.

### Implementation Notes

- **Point 4 deviated slightly from the original plan.** The plan called for "a new static node" for `out_of_scope`. In practice, `RouterNode.classify()`'s `out_of_scope` branch (point 3) already attaches a complete, schema-aware `AIMessage` and routes to the existing `next_step: "nonsense"` — so `fallback_node` just needed to stop calling the LLM, not gain new logic. It was changed to a no-op (`return {}`), matching the pre-existing `clarification_node` pattern exactly. This also fixed a second, previously-unnoticed bug: `fallback_node`'s old LLM call was stacking a **second, generic** "please rephrase" reply after the router's specific one on every `nonsense`/`out_of_scope` turn.
- **Point 6 is future-proofing, not a live bug fix.** Every current branch in `classify()` already sets `next_step` explicitly, so `reset_state["next_step"] = None` is always immediately overridden today. The value is in what happens if a future branch is added without setting it: routing fails loudly with a missing-edge error on the next turn, instead of silently replaying whatever `next_step` the previous turn left in the checkpoint.

### Tests Added

| File | Covers |
|---|---|
| `tests/test_router_node.py` | All `RouterNode.classify()` branches (greeting, out_of_scope, not-answerable, ambiguous, clean sql_query) reset `visualization`/`relevant_tables` correctly; `out_of_scope` routes safely even if `is_answerable` is mis-set |
| `tests/test_server_stream_response.py` | `build_stream_response_data()` only forwards `visualization` when `next_step == "end"` |
| `tests/test_graph_nodes.py` | `fallback_node` makes no LLM call and returns `{}`; `greeting_node` still calls the LLM |
| `tests/test_agent_graph_e2e.py` | Full 4-turn scenario through the real compiled graph + checkpointer: SQL chart → chit-chat → greeting → gibberish, asserting no stale chart and no unwanted LLM calls at each step |

Each fix was verified to actually catch the bug it targets by reverting the corresponding source change and re-running its test (all failed as expected — `KeyError`, `ImportError`, or `ValidationError` depending on the change — then passed again once restored).

### Status

✅ **Completed.** All 6 items implemented and tested; see GitHub issue #19.

---

## 23. 🐛 ROUND() on `double precision` Crashes Revenue-Share Pie Chart Queries

**Files:** `prompts/system_prompt.md`

### Observed Behaviour

User query *"Revenue share by brand this year"* generates:

```sql
SELECT brand,
ROUND(SUM(quantity * unit_price) * 100.0 / SUM(SUM(quantity * unit_price)) OVER (), 1) AS revenue_share
FROM sales
WHERE order_date >= date_trunc('year', CURRENT_DATE)
AND brand IS NOT NULL
GROUP BY brand
ORDER BY revenue_share desc;
```

This fails when executed against PostgreSQL:

```
ERROR: function round(double precision, integer) does not exist
HINT: No function matches the given name and argument types. You might need to add explicit type casts.
```

---

### Root Cause — A Verbatim Shipped Prompt Example, Not a Model Error

PostgreSQL only defines `round(numeric, integer)` — there is **no** `round(double precision, integer)` overload.

- `sales.quantity` is `BIGINT`, `sales.unit_price` is `DOUBLE PRECISION` (confirmed in `temp/db_schema.json:604,616`, `instructions/instruction.txt:131-132`, and `system_prompt.md:82-83` itself).
- `quantity * unit_price` → `double precision`. `SUM(double precision)` **stays** `double precision` — unlike `SUM(bigint)`, which PostgreSQL promotes to `numeric`.
- So the expression inside `ROUND(...)` ends up `double precision`, which has no two-argument `round()` overload, and the query errors.

This isn't the model improvising — the exact broken pattern is a few-shot example baked into the prompt itself:

- `system_prompt.md:1215-1225` — **"Revenue share by brand this year"** (essentially identical to the failing query)
- `system_prompt.md:1256-1265` — **"Revenue share by origin this year"** (identical bug, second copy)

Both were added in commit `bc1c405` (#15) and are still present on `HEAD`.

The other 7 pie-chart percentage examples in the same `### PIE CHART QUERIES` section (lines 1171, 1182, 1193, 1205, 1232, 1248, 1272) don't break, because they use `COUNT(*)`/`COUNT(DISTINCT ...)` — `bigint`, which `SUM()` promotes to `numeric` automatically. Only the two `SUM(quantity * unit_price)` variants touch a `double precision` column, and they're missing the `::NUMERIC` cast already used correctly elsewhere in the same file:

```sql
-- system_prompt.md:578-580 — correct pattern, already in use
SELECT ROUND(
    SUM(quantity * unit_price)::NUMERIC / COUNT(DISTINCT reference_no), 2
) AS avg_revenue_per_order
```

---

### Impact

| Dimension | Effect |
|---|---|
| **Correctness** | Any "revenue/sales share by X" pie-chart query fails outright with a SQL error instead of returning a chart |
| **Scope** | Affects exactly 2 of the 9 pie-chart few-shot examples — only the ones involving `SUM(quantity * unit_price)` — plus any new query the model writes by following that example |
| **Detectability** | Silent until a user asks a revenue-share question; not caught by any existing test, since prompt examples aren't executed against a real database in CI |

---

### Fix / TODO List

- [x] **`system_prompt.md:1218`** — cast both `SUM(quantity * unit_price)` occurrences to `::NUMERIC` in the "Revenue share by brand" example, matching the style at line 579
- [x] **`system_prompt.md:1259`** — same fix for "Revenue share by origin" (identical bug; fixed in the same edit as the brand example above)
- [x] **`system_prompt.md` SQL GENERATION RULES (lines 226-241)** — added rule 14: always cast aggregate expressions to `::NUMERIC` before passing them to `ROUND()` when the underlying column may be `double precision`/`real` (e.g. `unit_price`); `COUNT()`-based aggregates don't need it
- [x] **Audit pass** — grepped the rest of `system_prompt.md` for `ROUND(`/`round(` calls. Found and fixed a **third instance** at `system_prompt.md:1377` ("Show month-over-month revenue growth this year"): the CTE computed `SUM(quantity * unit_price) AS revenue` without a cast, and `revenue` then flowed through `LAG()`, subtraction, and division into `ROUND(..., 1)` at line 1385 — same root cause, since `double precision` propagates through arithmetic with a `numeric` literal (`numeric → float8` is an implicit cast in Postgres, but not the reverse) all the way to the `ROUND()` call. Fixed by casting at the source: `SUM(quantity * unit_price)::NUMERIC AS revenue` in the CTE. All other `ROUND(SUM(...))`/`ROUND(COUNT(...))` call sites in the file were confirmed already safe (either `COUNT()`-based, which `SUM()` promotes to `numeric` automatically, or already carrying an explicit `::NUMERIC` cast).
- [ ] *(Optional, separate follow-up)* Consider migrating `sales.unit_price` to `NUMERIC(12,2)` at the schema level to eliminate this class of bug entirely — requires a real migration + schema re-extraction + pgvector re-ingestion, out of scope for this fix

### Status

✅ **Implemented.** All required fixes (4 of 5 TODO items) are done; only the optional schema-migration follow-up remains open. See GitHub issue #21.

---

## 24. 🔍 `/stream` Doesn't Surface Intermediate Progress — Analysis & Plan

**Files:** `server.py`, `agent/graph.py`, `agent/subgraphs/sql_query/graph.py`, `agent/subgraphs/sql_query/nodes/*.py`, `agent/schema.py`

### Observed Behaviour

`POST /stream` sends exactly one SSE event per turn for SQL queries — the final `{content, visualization}` payload. There is no intermediate signal while the agent is calling the LLM, generating SQL, validating it, or executing it against PostgreSQL. For longer-running turns the client just waits with no feedback.

### Root Cause — 3 Layers, All Need to Change Together

**Layer 1 — `sql_query_node` hides the subgraph from the stream (`agent/graph.py:40-113`)**
It calls `sql_subgraph.invoke(subgraph_input)` — synchronous, blocking, and **no `config` is passed through**. LangGraph treats this as one opaque node. The subgraph itself was explicitly built "one action per node... for proper streaming support" (`subgraphs/sql_query/graph.py:39-43`), but that intent is defeated by the wrapping function: the parent graph has no visibility into `retrieve_schemas` → `generate_sql` → `validate_query` → `execute_query` finishing individually.

**Layer 2 — the parent stream call uses the wrong mode (`server.py:265-267`)**
`graph.astream(inputs, config, stream_mode="values")` only emits a snapshot when a *top-level* node finishes. Since `sql_query_node` is one top-level node that internally blocks until everything is done, exactly one snapshot appears for the entire SQL flow.

**Layer 3 — the event filter only forwards `AIMessage` chunks (`server.py:271-274`)**
`if "messages" in chunk and chunk["messages"]: ... if isinstance(last_message, AIMessage)`. Intermediate nodes don't append `AIMessage`s — they mutate state fields like `generated_sql`, `query_result`. Even a more granular stream would be silently dropped by this filter today.

There's also no schema for a "progress" event — the SSE payload (`build_stream_response_data`, `server.py:200-214`) is shaped only for the final answer, with no `type`/`stage` discriminator a client could use to tell a progress ping apart from the real answer.

### Feasibility

**Confirmed possible — no rewrite required.** LangGraph natively supports nested-subgraph streaming (`stream_mode="updates", subgraphs=True`) and fine-grained node/LLM events (`astream_events`). The code just isn't wired to use either; all 4 subgraph nodes are already sync functions with no async I/O wiring (`retrieve_schemas_node` already accepts `config: RunnableConfig`, so the plumbing point exists).

### TODO List (Deferred)

**Subgraph invocation (`agent/graph.py`)**
- [ ] Make `sql_query_node` `async def` and accept `config: RunnableConfig`
- [ ] Replace `sql_subgraph.invoke(subgraph_input)` with `await sql_subgraph.ainvoke(subgraph_input, config=config)` (or `astream`) so callbacks/run context propagate into the subgraph instead of being swallowed
- [ ] Convert the 4 subgraph node functions (`retrieve_schemas_node`, `generate_sql_node`, `validate_query_node`, `execute_query_node`) to `async def` where they call I/O (LLM calls, DB calls) so they play well with `ainvoke`/`astream`

**Stream mode (`server.py`)**
- [ ] Switch `graph.astream(...)` to `stream_mode="updates"` (or add `astream_events()` as a parallel path) so per-node completions are visible, not just full-state snapshots
- [ ] Define a node-name → human-readable progress message map (e.g. `retrieve_schemas` → "Looking up relevant tables…", `generate_sql` → "Generating SQL query…", `validate_query` → "Validating query…", `execute_query` → "Running query…")
- [ ] Update `event_generator()` to emit a progress event on each relevant node completion, not just when an `AIMessage` appears

**SSE contract**
- [ ] Add a `type` discriminator to the SSE payload, e.g. `{"type": "progress", "stage": "generate_sql", "message": "..."}` vs `{"type": "final", "content": ..., "visualization": ...}`, so clients can tell pings apart from the answer
- [ ] Update the `/stream` endpoint's OpenAPI `description` to describe both event types
- [ ] Flag to the frontend team that this is a breaking change to the SSE event shape — any existing client parsing `{content, visualization}` directly needs to start checking `type` first

**Out of scope / optional**
- [ ] Cancellation/timeout handling per stage if a step (e.g. LLM call) hangs
- [ ] Retry-loop visibility: `execute_query` can route back to `generate_sql` (`routes.py` / `route_after_execution`) — decide whether a retry should re-emit "Generating SQL…" or a distinct "Retrying…" message

### Status

⏸️ **Deferred.** Analysis and plan only — not yet implemented. Project is currently focused on the POC; revisit once the POC is validated.

---

---

## 25. 📝 Visualization Analysis — Feature Plan

**Files affected:** `agent/state.py`, `agent/graph.py`, `server.py`, `prompts/analysis_prompt.md` (new), `tests/test_agent_graph_e2e.py`

### Problem Statement

Every SQL turn currently returns a visualization object (chart type + data) but no natural-language commentary on what the numbers mean. The client receives a boilerplate one-liner from `_generate_brief_content()` (e.g. `"Here's the delivery method split distribution:"`) and a heuristic template in `visualization.summary` (e.g. `"Distribution across 2 categories."`). Neither tells the user what the data actually says.

---

### Root Cause / Current State

The data flow today ends without surfacing any insight:

1. `sql_query_node` (`agent/graph.py:40`) calls `sql_subgraph.invoke()`, which returns a dict containing `explanation`, `query_result`, `generated_sql`, and other fields.
2. `sql_query_node` calls `visualizer_node`, which uses pure heuristics to pick a chart type and produces a `visualization` dict. `visualization.summary` is always a template string (`"Distribution across 2 categories."`).
3. `sql_query_node` calls `_generate_brief_content(visualization)`, producing a one-liner like `"Here's the delivery method split distribution:"`. This becomes the `AIMessage` content stored in state.
4. `build_stream_response_data` in `server.py` assembles the SSE payload as `{"content": "<one-liner>", "visualization": {...}}`.
5. The `explanation` field from the SQL subgraph (LLM-generated, e.g. `"Distribution of delivery method."`) is **never declared on `AgentState`** and **never included in the SSE payload** — it silently disappears after `sql_query_node` completes.
6. `format_sql_response_with_visualization` and `format_sql_response` in `graph.py` are dead code — never called from the live path.
7. `visualizer.py:48` already has a `# TODO: apply intelligent heuristics using LLMs` comment confirming this gap was always known.

---

### `/stream` Response Body — Before vs After

#### Before (current)

```json
{
  "content": "Here's the delivery method split distribution:",
  "visualization": {
    "type": "pie_chart",
    "title": "Delivery Method Split",
    "data": [
      {"delivery_method": "DELIVERY", "percentage": 94.2},
      {"delivery_method": "PICKUP",   "percentage": 5.8}
    ],
    "columns": ["delivery_method", "percentage"],
    "config": { "category": "delivery_method", "value": "percentage" },
    "summary": "Distribution across 2 categories.",
    "row_count": 2
  }
}
```

`summary` is always a boilerplate template — never real insight.

#### After (with analysis field)

```json
{
  "content": "Here's the delivery method split distribution:",
  "visualization": {
    "type": "pie_chart",
    "title": "Delivery Method Split",
    "data": [
      {"delivery_method": "DELIVERY", "percentage": 94.2},
      {"delivery_method": "PICKUP",   "percentage": 5.8}
    ],
    "columns": ["delivery_method", "percentage"],
    "config": { "category": "delivery_method", "value": "percentage" },
    "summary": "Distribution across 2 categories.",
    "row_count": 2
  },
  "analysis": "DELIVERY accounts for 94.2% of all shipments, making it the dominant fulfilment method. PICKUP represents only 5.8% of orders, suggesting most customers prefer home delivery. This split indicates a strong preference for doorstep delivery across your customer base."
}
```

#### Non-SQL turns (greetings, refusals, clarifications)

```json
{
  "content": "Hello! How can I help you today?",
  "visualization": null,
  "analysis": null
}
```

**Frontend contract:**
- `analysis` is always present — either a non-empty string or `null`.
- `analysis` is non-null **only** when `visualization` is also non-null (both guarded by the same `went_through_sql_node` flag in `build_stream_response_data`).
- `content` remains the short header line — `analysis` is the new richer companion field, not a replacement.

---

### P1 — Must-Do (minimum to ship)

| # | File | Change |
|---|------|--------|
| P1-A | `agent/state.py` | Add `analysis: Optional[str]` to `AgentState`. Without this declaration the field is not checkpointed and not type-safe. |
| P1-B | `agent/graph.py` | After `visualizer_node` runs inside `sql_query_node`, call `fast_llm` (already module-level) with a structured prompt (user query + chart type + chart title + row count + first 3-5 data rows). Return the resulting string as `"analysis"` in the `sql_query_node` return dict alongside `"visualization"`. Also add `"analysis": None` to both early-return error branches of `sql_query_node` to prevent state leakage from prior turns. |
| P1-C | `server.py` | Add `"analysis": accumulated_state.get("analysis") if went_through_sql_node else None` to `build_stream_response_data`. Same gate as `visualization` — no extra logic needed. |
| P1-D | `tests/test_agent_graph_e2e.py` | Extend the Turn 1 assertion to check `state1.get("analysis")` is a non-empty string. Add negative checks on Turns 2, 3, 4 that `state.get("analysis")` is `None`. The `mock_fast_llm` fixture already patches `graph.fast_llm` — set its return value for the analysis call on Turn 1. |

**Minimal analysis prompt template (to embed in `graph.py` or in `analysis_prompt.md`):**

```
You are a data analyst. In 2-3 sentences, explain what the following result means in plain English for a business user. Be specific about numbers where available. Do not repeat the chart title verbatim.

User question: {user_query}
Chart type: {viz_type}
Chart title: {viz_title}
Row count: {row_count}
Top data:
{first_3_rows_as_json}
```

**LLM choice:** `fast_llm` (`ModelType.FAST`) — already instantiated at module level in `graph.py:13`. Used here for the same reason it's used for routing: low-latency, lightweight generation task.

---

### P2 — Should-Do

| # | File | Change |
|---|------|--------|
| P2-A | `agent/graph.py` | Wrap the `fast_llm` analysis call in `try/except`. On failure, fall back to `subgraph_result.get("explanation", "")` (the single-sentence SQL description from `SQLGeneration`). Prevents the feature from erroring if the LLM call fails. |
| P2-B | new `prompts/analysis_prompt.md` | Move the analysis prompt out of Python into the `prompts/` directory, matching the existing convention (`system_prompt.md`, `router_prompts.md`, `schema_to_text_prompts.md`). Load it via `get_system_prompt` / `prompt_loader.py`. Makes prompt tuning possible without touching Python code. |
| P2-C | `agent/graph.py` | Verify that the `analysis` reset (`"analysis": None`) in the non-SQL branches prevents prior-turn values from leaking — same class of bug that issue #19 fixed for `visualization`. Covered by the P1-D negative assertions. |

---

### P3 — Nice-to-Have

| # | What |
|---|------|
| P3-A | Pass first 3-5 actual data rows as compact JSON into the analysis prompt (included in P1-B above). Lets the LLM cite specific numbers (`"DELIVERY accounts for 94.2%..."`) instead of vague summaries. Cap at 5 rows; truncate long string values to avoid inflating token usage. |
| P3-B | Update `test_server_stream_response.py` (if it exists) to assert the `analysis` key is present in the emitted SSE JSON (null or string) — machine-checked API contract. |
| P3-C | Consider replacing the boilerplate `_generate_brief_content()` one-liner with the analysis text as `content` directly. Cleaner: one field for the human-readable response instead of two. Tradeoff: current `content` is instant (template, no LLM call); replacing it means the chat bubble is empty until the analysis round-trip completes. |

---

### Key File Locations

| File | Relevant Lines / Notes |
|------|----------------------|
| `src/ai_agentic_chatbot/agent/state.py` | `AgentState` TypedDict — add `analysis: Optional[str]` |
| `src/ai_agentic_chatbot/agent/graph.py` | `sql_query_node` (line 40); `fast_llm` module-level instance (line 13); `_generate_brief_content` (line 131) |
| `src/ai_agentic_chatbot/agent/nodes/visualizer.py` | `# TODO: apply intelligent heuristics using LLMs` comment (line 48); `_create_payload` where `summary` is set (line 294) |
| `src/ai_agentic_chatbot/agent/subgraphs/sql_query/state.py` | `SQLSubgraphState` — `explanation: Optional[str]` (line 19); available as fallback |
| `src/ai_agentic_chatbot/agent/subgraphs/sql_query/nodes/generate_sql.py` | `SQLGeneration` Pydantic model — `explanation` field (line 21) |
| `src/ai_agentic_chatbot/server.py` | `build_stream_response_data` (line 200) — only place SSE payload is assembled |
| `src/ai_agentic_chatbot/prompts/` | All prompt files live here; no analysis prompt yet |
| `tests/test_agent_graph_e2e.py` | `SQL_SUBGRAPH_RESULT` fixture (line 36) already has `"explanation"` field confirming it's available but untested downstream |

### Status

✅ **Completed.** Analysis and TODO list complete — implemented.

---

---

## 26. 🔐 User Authentication Module — Implementation Plan

**Files affected:** `src/ai_agentic_chatbot/auth/` (new package), `src/ai_agentic_chatbot/infrastructure/database.py` (new), `src/ai_agentic_chatbot/infrastructure/db_depency.py`, `src/ai_agentic_chatbot/server.py`, `pyproject.toml`, `alembic/` (new), `.env`

### Problem Statement

The application currently has no authentication layer. All API endpoints — including the main `/stream` chat endpoint and the admin schema pipeline (`/schemaJson`, `/schemaText`, `/ingest`) — are fully public. Any client with network access can call them without credentials. A user login module is required to:

- Allow users to log in via username + password from the UI
- Issue JWT access tokens for authenticated sessions
- Protect all sensitive endpoints behind token validation

---

### Architecture Overview

```
POST /auth/login  (public)
        │  OAuth2PasswordRequestForm (form-encoded)
        ▼
  authenticate_user()
        │  passlib bcrypt verify
        ▼
  create_access_token()
        │  python-jose JWT, signed with JWT_SECRET_KEY
        ▼
  Token { access_token, token_type: "bearer" }

─────────────────────────────────────────────

Protected Endpoint (e.g. POST /stream)
        │  Authorization: Bearer <token>
        ▼
  get_current_active_user()  ← FastAPI Depends
        │  decode_access_token() → username
        │  get_user_by_username(db) → User ORM
        ▼
  Endpoint handler runs
```

---

### New Package Structure

```
src/ai_agentic_chatbot/auth/
├── __init__.py
├── dependencies.py      # get_current_user, get_current_active_user, get_current_superuser
├── jwt_utils.py         # create_access_token, decode_access_token
├── models.py            # User SQLAlchemy ORM model (users table)
├── password.py          # hash_password, verify_password (bcrypt)
├── repository.py        # get_user_by_username, get_user_by_email, create_user
├── router.py            # POST /auth/login, POST /auth/register, GET /auth/me
├── schemas.py           # UserCreate, UserResponse, Token, TokenData (Pydantic v2)
├── seed.py              # First superuser bootstrap script
└── service.py           # authenticate_user, create_user_account

src/ai_agentic_chatbot/infrastructure/
└── database.py          # SQLAlchemy DeclarativeBase → Base (new — shared by all ORM models)
```

---

### Database — `users` Table

| Column | Type | Constraints |
|---|---|---|
| `id` | `BigInteger` | `Identity()`, primary key |
| `username` | `String(100)` | unique, not null, indexed |
| `email` | `String(255)` | unique, not null |
| `hashed_password` | `Text` | not null |
| `is_active` | `Boolean` | not null, server default `true` |
| `is_superuser` | `Boolean` | not null, server default `false` |
| `created_at` | `DateTime(timezone=True)` | not null, server default `func.now()` |
| `updated_at` | `DateTime(timezone=True)` | not null, server default + `onupdate` |

Migration managed by **Alembic** (`alembic upgrade head`).

---

### Auth Endpoints

| Method | Path | Auth Required | Description |
|---|---|---|---|
| `POST` | `/auth/login` | Public | Form-encoded `username` + `password` → returns `Token` |
| `POST` | `/auth/register` | Superuser only | JSON `UserCreate` body → creates new user |
| `GET` | `/auth/me` | Active user | Returns `UserResponse` for current session |

---

### Endpoint Protection

| Endpoint | Protection Level |
|---|---|
| `POST /stream` | `get_current_active_user` |
| `GET /schemaJson` | `get_current_superuser` |
| `GET /schemaText` | `get_current_superuser` |
| `GET /ingest` | `get_current_superuser` |
| `GET /health` | Public (liveness probe) |
| `GET /db-health` | Public (internal probe) |

---

### Dependencies Added

| Package | Purpose |
|---|---|
| `python-jose[cryptography]` | JWT encode/decode |
| `passlib[bcrypt]` | bcrypt password hashing |
| `python-multipart` | FastAPI `OAuth2PasswordRequestForm` body parsing |
| `alembic` | Database migration runner |

---

### Configuration

New environment variables (`.env` only — never in `config.yaml`):

```
JWT_SECRET_KEY=<generate: python -c "import secrets; print(secrets.token_hex(32))">
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

# Only needed for initial seed run
SEED_USERNAME=
SEED_EMAIL=
SEED_PASSWORD=
```

---

### Bootstrap Flow (first deployment)

```
1. alembic upgrade head          → creates users table
2. Set SEED_* env vars
3. python -m ai_agentic_chatbot.auth.seed   → inserts first superuser
4. POST /auth/login              → get access token
5. POST /auth/register           → create additional users (superuser token required)
```

---

### Frontend Integration Notes

- Login endpoint uses `application/x-www-form-urlencoded` (NOT JSON):
  ```javascript
  const form = new URLSearchParams({ username, password });
  fetch('/auth/login', { method: 'POST', body: form });
  ```
- Store token in `sessionStorage` (cleared on tab close — better XSS posture than `localStorage`)
- Attach `Authorization: Bearer <token>` to every subsequent API call
- Intercept `401` responses globally → redirect to login page
- On page reload, call `GET /auth/me` to verify token is still valid

---

### Key Design Decisions

| Decision | Rationale |
|---|---|
| JWT (stateless) over sessions | No server-side session store needed; works across multiple instances |
| `get_auth_db()` as a generator with `finally: session.close()` | Existing `get_db_session()` never closes the session — auth introduces the correct FastAPI `Depends` generator pattern |
| No `auth` in `config.yaml` | JWT secret must not be committed; reads from `.env` via `os.environ.get()` |
| `POST /auth/register` requires superuser | Prevents open self-registration; new users created only by an admin |
| Alembic for migrations | Consistent with SQLAlchemy ecosystem; autogenerate detects model changes |

---

### TODO List

| Step | Phase | File | Action |
|---|---|---|---|
| 1 | Dependencies | `pyproject.toml` | Add `python-jose`, `passlib`, `python-multipart`, `alembic` |
| 2 | Config | `.env` / `.env.example` | Add `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` |
| 3 | Migration | `alembic/` | `alembic init alembic`; configure `alembic.ini` |
| 4 | ORM Base | `infrastructure/database.py` | Create `Base = DeclarativeBase()` |
| 5 | User Model | `auth/models.py` | `User` ORM model mapping to `users` table |
| 6 | Alembic env | `alembic/env.py` | Import `Base` + `auth.models`; set `target_metadata`; read DB URL from env |
| 7 | Migration | `alembic/versions/` | `alembic revision --autogenerate` → `alembic upgrade head` |
| 8 | Password | `auth/password.py` | `hash_password`, `verify_password` via `passlib` |
| 9 | JWT | `auth/jwt_utils.py` | `create_access_token`, `decode_access_token` via `python-jose` |
| 10 | Schemas | `auth/schemas.py` | `UserCreate`, `UserResponse`, `Token`, `TokenData` (Pydantic v2) |
| 11 | Repository | `auth/repository.py` | `get_user_by_username`, `get_user_by_email`, `create_user` |
| 12 | Service | `auth/service.py` | `authenticate_user`, `create_user_account` |
| 13 | DB Dep | `infrastructure/db_depency.py` | Add `get_auth_db()` generator with `yield` + `finally` |
| 14 | FastAPI Dep | `auth/dependencies.py` | `get_current_user`, `get_current_active_user`, `get_current_superuser` |
| 15 | Router | `auth/router.py` | `POST /auth/login`, `POST /auth/register`, `GET /auth/me` |
| 16 | Register | `server.py` | `app.include_router(auth_router)` |
| 17 | Secure APIs | `server.py` | Add `Depends(...)` to `/stream`, `/schemaJson`, `/schemaText`, `/ingest` |
| 18 | Seed | `auth/seed.py` | First superuser bootstrap script |
| 19 | Package | `auth/__init__.py` | Empty package marker |
| 20 | Frontend | UI code | Form-encoded login, `sessionStorage`, auth header, 401 handler |

### Status

✅ **Completed.** All 19 implementation steps done and verified end-to-end (2026-06-26).

---

---

## 27. 🔑 JWT Refresh Token System — Design & Implementation Plan

### Problem Statement

The current access token has a 60-minute TTL. When it expires, the user must re-enter their credentials. For a chatbot that runs across a workday session, this creates unacceptable UX friction. A refresh token system allows the client to silently obtain a new access token without prompting the user — extending the effective session lifetime to days while keeping the access token short-lived.

---

### Two-Token Architecture

```
POST /auth/login
    └─► { access_token (15–60 min), refresh_token (7–14 days) }

Every protected API call:
    Authorization: Bearer <access_token>

Access token expires:
    POST /auth/refresh  { refresh_token: "<token>" }
    └─► { access_token (new), refresh_token (new) }

User logs out:
    POST /auth/logout  { refresh_token: "<token>" }
    └─► HTTP 204 (token revoked in DB)
```

The access token remains a short-lived JWT verified statlessly. The refresh token is an opaque random string always validated against the database — never a JWT.

---

### Core Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Refresh token format | Opaque random string (`secrets.token_urlsafe(64)`) | DB-validated anyway; embedding claims in a JWT adds nothing |
| Storage | PostgreSQL table (`refresh_tokens`) | No Redis needed; single indexed lookup is fast enough at this scale |
| What is stored | SHA-256 hash of the raw token | Raw token goes to client only; if DB is exfiltrated, attacker gets hashes not usable tokens |
| Token transport | Response body (JSON) | No browser frontend yet; HttpOnly cookies add CSRF complexity with no XSS benefit for API/mobile clients |
| Rotation | Mandatory on every refresh | Each `/auth/refresh` invalidates the old token and issues a new pair |
| Replay detection | Token family model | Detecting a used token → revoke entire family → force re-login |

---

### Token Family Model (Replay Attack Detection)

Every login creates a new `family_id` (UUID). All refresh tokens issued from that login share the same `family_id`.

```
Login ──► access_token_1 + refresh_token_A  (family_id = X)
              │
              ▼ /auth/refresh with refresh_token_A
         mark A used=True
         ──► access_token_2 + refresh_token_B  (family_id = X)
              │
              ▼ attacker replays refresh_token_A (already used=True)
         REPLAY DETECTED
         ──► revoke all tokens where family_id = X
         ──► 401 to both legitimate user and attacker
         ──► user must log in again
```

This design is **self-reporting theft detection**: using a stolen token alerts the system.

---

### Database Schema — `refresh_tokens` Table

```python
class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id:         UUID (PK, random)        # UUID4 — avoids sequential enumeration
    user_id:    BigInt FK → users.id     # CASCADE DELETE; indexed for family revocation
    token_hash: String(64), unique       # SHA-256 hex of raw token (always 64 chars)
    family_id:  UUID, indexed            # Groups all tokens from one login session
    used:       Boolean (default False)  # True = this token was already rotated away
    revoked:    Boolean (default False)  # True = force-revoked (logout / replay detected)
    expires_at: DateTime(tz)             # Hard expiry; used for DB cleanup
    created_at: DateTime(tz)             # Audit trail
```

**`used` vs `revoked`:**
- `used = True` + `revoked = False` → normal rotation (legitimate previous use)
- `used = True` + `revoked = True` → replay detected (security event)
- `used = False` + `revoked = True` → explicit logout or admin revocation

---

### New Endpoints

#### `POST /auth/login` (modified)
Currently returns `{ access_token, token_type }`. Must also generate a refresh token row and return `refresh_token` in the response.

**New response:**
```json
{ "access_token": "<jwt>", "refresh_token": "<opaque>", "token_type": "bearer" }
```

---

#### `POST /auth/refresh`

**Request body (JSON):**
```json
{ "refresh_token": "<opaque_token_string>" }
```

**Logic:**
1. Hash the incoming token: `SHA-256(raw_token).hexdigest()`
2. Look up row by `token_hash`
3. Not found → 401 "Invalid refresh token"
4. `revoked == True` → 401 "Token has been revoked"
5. `used == True` → revoke entire `family_id` → 401 "Token reuse detected" *(replay attack)*
6. `expires_at < now()` → 401 "Refresh token expired"
7. Mark current row `used = True`
8. Generate new raw token, store its hash with same `family_id`, `expires_at = now() + 7 days`
9. Issue new `create_access_token({"sub": user.username})`

**Response (200):**
```json
{ "access_token": "<new_jwt>", "refresh_token": "<new_opaque>", "token_type": "bearer" }
```

---

#### `POST /auth/logout`

**Request body (JSON):**
```json
{ "refresh_token": "<opaque_token_string>" }
```

**Logic:**
1. Hash incoming token
2. Look up row by `token_hash`
3. If found and not revoked → set `revoked = True`, commit
4. Always return **HTTP 204** regardless (do not leak whether token existed)

---

### Files Changed

| File | Change |
|---|---|
| `auth/models.py` | Add `RefreshToken` ORM class |
| `auth/schemas.py` | Add `refresh_token: str \| None = None` to `Token`; add `RefreshRequest` schema |
| `auth/repository.py` | Add `create_refresh_token`, `get_refresh_token_by_hash`, `mark_token_used`, `revoke_family`, `revoke_token` |
| `auth/jwt_utils.py` | Add `generate_refresh_token()` and `hash_token()` helpers |
| `auth/router.py` | Modify `login`; add `POST /auth/refresh` and `POST /auth/logout` |
| `alembic/env.py` | Import `RefreshToken` so autogenerate detects the new table |

---

### Migration Steps

1. Add `RefreshToken` to `auth/models.py`
2. Import it in `alembic/env.py`
3. Run `alembic revision --autogenerate -m "create refresh_tokens table"`
4. Inspect generated file — confirm only `create_table('refresh_tokens', ...)` with no drops
5. Run `alembic upgrade head`

---

### Security Notes

- **Never store the raw token** — only `hashlib.sha256(raw.encode()).hexdigest()`
- **SHA-256 without salt** is safe here because `secrets.token_urlsafe(64)` produces 512 bits of entropy — rainbow tables are infeasible
- **Expired token cleanup** — run `DELETE FROM refresh_tokens WHERE expires_at < NOW()` periodically (at login time or via a cron job) to prevent table bloat
- **Revisit HttpOnly cookie** if a browser SPA is added as a client — the response body approach is correct only while clients are API/mobile/server-to-server

---

### TODO List

| Step | File | Action |
|---|---|---|
| 20 | `auth/models.py` | Add `RefreshToken` ORM model |
| 21 | `alembic/env.py` | Import `RefreshToken` for autogenerate |
| 22 | `alembic/versions/` | Generate + apply `create_refresh_tokens_table` migration |
| 23 | `auth/jwt_utils.py` | Add `generate_refresh_token()` and `hash_token()` |
| 24 | `auth/schemas.py` | Update `Token` schema; add `RefreshRequest` schema |
| 25 | `auth/repository.py` | Add refresh token CRUD functions |
| 26 | `auth/router.py` | Modify `login`; add `/auth/refresh` and `/auth/logout` |
| 27 | `.env` / `.env.example` | Add `JWT_REFRESH_TOKEN_EXPIRE_DAYS=7` |

### Status

✅ **Completed.** All 8 implementation steps (Steps 20–27) done and verified end-to-end (2026-06-27).

---

---

## 28. Alembic vs Docker — Migration System Clash Analysis

> ⚠️ **Priority: HIGH** — Must be resolved before any production deployment involving schema changes. App will crash on a fresh DB or after new migrations if left unaddressed.

### Background

Database migrations are managed by **Alembic** (`alembic/`, `alembic.ini`). The application is containerised via **Docker** (`Dockerfile`, `entrypoint.sh`, `docker-compose.yml`, `docker-compose.prod.yml`). These two systems currently do not talk to each other, creating three critical gaps.

---

### Issue 1 — CRITICAL: `alembic/` is not copied into the Docker image

**What the Dockerfile copies:**

```
COPY src/           → app source code
COPY certs/         → TLS certificates
COPY config.example.yaml
COPY entrypoint.sh
```

`alembic/` and `alembic.ini` are **not `COPY`-ed**. The `alembic` CLI cannot be invoked from inside any running container — the files simply do not exist there.

**Impact:** Migrations must be run manually from the developer's machine before every deploy. This is invisible to the Docker workflow and trivially easy to forget, especially in CI/CD pipelines.

---

### Issue 2 — CRITICAL: No migration step runs before the application starts

`entrypoint.sh` flow:

```
1. Validate required env vars
2. Wait for PostgreSQL (TCP probe — up to 60s)
3. exec uvicorn   ← app starts immediately, no alembic upgrade head
```

There is no `alembic upgrade head` between steps 2 and 3. On a fresh database — or after any new migration is added (e.g., `users` table, `refresh_tokens` table) — the container starts, uvicorn boots, and the first authenticated request hits a `relation "users" does not exist` error.

**Affected migrations already in the repo:**

| Migration | Table |
|---|---|
| `afab544e8c58` | `users` |
| `65c0b5446016` | `refresh_tokens` |

---

### Issue 3 — MODERATE: `prepend_sys_path` is wrong for local Alembic runs

`alembic.ini` line 21:

```ini
prepend_sys_path = .
```

This adds the project root (`.`) to `sys.path`. But the `ai_agentic_chatbot` package lives under `src/`, so the import in `alembic/env.py`:

```python
from ai_agentic_chatbot.infrastructure.database import Base
import ai_agentic_chatbot.auth.models
```

...will raise `ModuleNotFoundError` unless `PYTHONPATH=src` is set in the shell before invoking `alembic`. The Dockerfile sets `ENV PYTHONPATH=/app/src`, but since `alembic/` is not in the image, that env var never helps.

**Result:** Developers must remember to run `$env:PYTHONPATH="src"; alembic upgrade head` (PowerShell) or `PYTHONPATH=src alembic upgrade head` (bash) — an undocumented requirement.

---

### Fix Plan

#### Fix 1 — Copy alembic files into the Docker image (`Dockerfile`)

```dockerfile
# After the COPY src/ line:
COPY alembic/ ./alembic/
COPY alembic.ini ./
```

#### Fix 2 — Run migrations in entrypoint (`entrypoint.sh`)

Add one line between the DB-wait block and `exec "$@"`:

```sh
echo "Running database migrations..."
alembic upgrade head
```

Full corrected entrypoint flow:
```
1. Validate env vars
2. Wait for PostgreSQL
3. alembic upgrade head   ← NEW
4. exec uvicorn
```

This makes every container start idempotent — Alembic detects if the schema is already current and exits instantly (no-op), so the overhead on restarts is negligible.

#### Fix 3 — Fix `prepend_sys_path` in `alembic.ini`

```ini
# Before:
prepend_sys_path = .

# After:
prepend_sys_path = src
```

This allows `alembic` commands to work from the project root without any manual `PYTHONPATH` export.

---

### Summary Table

| # | Issue | Severity | Fix Location |
|---|---|---|---|
| 1 | `alembic/` and `alembic.ini` not in Docker image | **Critical** | `Dockerfile` — add two `COPY` lines |
| 2 | No `alembic upgrade head` before uvicorn starts | **Critical** | `entrypoint.sh` — add one line after DB wait |
| 3 | `prepend_sys_path = .` breaks local `alembic` CLI | **Moderate** | `alembic.ini` — change `.` to `src` |

---

### TODO List

| Step | File | Action |
|---|---|---|
| 1 | `Dockerfile` | Add `COPY alembic/ ./alembic/` and `COPY alembic.ini ./` after the `COPY src/` line |
| 2 | `entrypoint.sh` | Add `alembic upgrade head` after the PostgreSQL wait block, before `exec "$@"` |
| 3 | `alembic.ini` | Change `prepend_sys_path = .` to `prepend_sys_path = src` |
| 4 | Manual test | Build image, start container against a fresh DB, confirm tables are created before uvicorn accepts traffic |

### Status

Pending — analysis recorded 2026-06-27. No code changes made yet.

---

---

## 29. User Prompt Management — Implementation Plan ✅ COMPLETED

Three related features: password self-service, prompt audit logging, and per-user daily limits. Features 1 and 2 are independent; Feature 3 depends on Feature 2.

```
Feature 1 (password update)  — no DB schema change, ship independently
     ↓
Feature 2 (save prompts)     — new prompt_logs table (migration required)
     ↓
Feature 3 (daily limit)      — new column on users + reads prompt_logs
```

---

### Codebase Baseline (as of 2026-06-27)

**User ORM columns (`auth/models.py`):** `id`, `username`, `email`, `hashed_password`, `is_active`, `is_superuser`, `created_at`, `updated_at` — no usage-tracking fields.

**`/stream` handler (`server.py:244`):** Already has `current_user: User = Depends(get_current_user)` and `stream_request: StreamRequest` in scope. `current_user` is currently unused beyond being an auth gate.

**Existing migrations:** `afab544e8c58` (users table) → `65c0b5446016` (refresh_tokens table).

**No password update path exists** anywhere in `repository.py`, `service.py`, or `router.py`.

---

### Feature 1 — `PATCH /auth/password` (Self-Service Password Update)

Username is taken from the authenticated JWT — not the request body. The caller must supply their current password to prove identity before setting a new one.

| Step | File | Action |
|---|---|---|
| 1.1 | `auth/schemas.py` | Add `PasswordUpdateRequest(BaseModel)` — fields: `current_password: str`, `new_password: str` (enforce min length 8). Add `PasswordUpdateResponse(BaseModel)` — field: `message: str`. |
| 1.2 | `auth/repository.py` | Add `update_user_password(db, user: User, new_hashed: str) -> User` — sets `user.hashed_password = new_hashed`, calls `db.commit()`, `db.refresh(user)`, returns user. `updated_at` auto-fires via ORM `onupdate`. |
| 1.3 | `auth/service.py` | Add `change_password(db, current_user, current_password, new_password)` — calls `verify_password(current_password, current_user.hashed_password)`, raises `HTTP 400` if wrong, calls `hash_password(new_password)`, calls `update_user_password`. |
| 1.4 | `auth/router.py` | Add `PATCH /auth/password` route — protected by `get_current_user`. Body: `PasswordUpdateRequest`. Calls `service.change_password(...)`. Returns `PasswordUpdateResponse`. |

---

### Feature 2 — Save Prompts to DB on Every `/stream` Call

New `prompt_logs` table stores each prompt with a FK to the submitting user and the conversation thread.

#### 2a. New ORM Model (`auth/models.py`)

Add `PromptLog` class to the existing models file:

| Column | Type | Constraints |
|---|---|---|
| `id` | `BigInteger` | PK, autoincrement |
| `user_id` | `BigInteger` | FK → `users.id` ON DELETE CASCADE, indexed |
| `thread_id` | `String(255)` | not null |
| `prompt_text` | `Text` | not null |
| `created_at` | `DateTime(timezone=True)` | not null, server_default `now()` |

#### 2b. Migration

| Step | File | Action |
|---|---|---|
| 2.1 | `auth/models.py` | Add `PromptLog` ORM class (columns above). |
| 2.2 | `alembic/versions/` | Run `alembic revision --autogenerate -m "create prompt_logs table"`. Inspect output — confirm only `create_table('prompt_logs', ...)` with no drops. Run `alembic upgrade head`. |

#### 2c. Repository + Handler

| Step | File | Action |
|---|---|---|
| 2.3 | `auth/repository.py` | Add `create_prompt_log(db, user_id: int, thread_id: str, prompt_text: str) -> PromptLog` — creates and commits a `PromptLog` row. |
| 2.4 | `server.py` | Inject `db: Session = Depends(get_auth_db)` into the `/stream` handler (alongside the existing `current_user` dependency). Before invoking the LangGraph agent, call `create_prompt_log(db, current_user.id, stream_request.thread_id, stream_request.messages[-1].content)`. |

---

### Feature 3 — Configurable Daily Prompt Limit per User

`daily_prompt_limit = 0` means unlimited. The limit check happens in `/stream` before the graph is invoked, so no LLM tokens are burned on rejected requests.

**Requires Feature 2** (`prompt_logs` table must exist before this feature can query it).

#### 3a. Schema Change

| Step | File | Action |
|---|---|---|
| 3.1 | `auth/models.py` | Add `daily_prompt_limit` column to `User` — `Column(Integer, nullable=False, server_default="0")`. |
| 3.2 | `alembic/versions/` | Run `alembic revision --autogenerate -m "add daily_prompt_limit to users"`. Inspect — confirm only `add_column('users', Column('daily_prompt_limit', Integer, ...))`. Run `alembic upgrade head`. |
| 3.3 | `auth/schemas.py` | Add `daily_prompt_limit: int = 0` to `UserResponse` so `GET /auth/me` exposes the field. |

#### 3b. Limit Check Logic

| Step | File | Action |
|---|---|---|
| 3.4 | `auth/repository.py` | Add `count_prompts_today(db, user_id: int) -> int` — queries `PromptLog` where `user_id = user_id` AND `created_at >= today 00:00 UTC`. Returns integer count. |
| 3.5 | `server.py` | In `/stream`, after saving the prompt log (Feature 2 step 2.4), add: if `current_user.daily_prompt_limit > 0` and `count_prompts_today(db, current_user.id) > current_user.daily_prompt_limit`, raise `HTTP 429` with detail `"Daily prompt limit reached"`. |

#### 3c. Admin Endpoint to Set Limits

| Step | File | Action |
|---|---|---|
| 3.6 | `auth/schemas.py` | Add `PromptLimitUpdateRequest(BaseModel)` — field: `daily_prompt_limit: int` (must be >= 0). |
| 3.7 | `auth/repository.py` | Add `update_user_prompt_limit(db, user: User, limit: int) -> User` — sets `user.daily_prompt_limit = limit`, commits. |
| 3.8 | `auth/router.py` | Add `PATCH /auth/users/{username}/limit` route — superuser-only (guard: `if not current_user.is_superuser: raise HTTP 403`). Looks up target user by username, calls `update_user_prompt_limit`. Returns updated `UserResponse`. |

---

### Full Execution Order

```
Step 1.1 → 1.2 → 1.3 → 1.4   (Feature 1 — password update, no migration)
     ↓
Step 2.1 → 2.2 (migration)    (Feature 2a — create prompt_logs table)
     ↓
Step 2.3 → 2.4                (Feature 2b — repository + hook into /stream)
     ↓
Step 3.1 → 3.2 (migration)    (Feature 3a — add daily_prompt_limit to users)
     ↓
Step 3.3 → 3.4 → 3.5          (Feature 3b — limit check in /stream)
     ↓
Step 3.6 → 3.7 → 3.8          (Feature 3c — admin limit-setting endpoint)
```

### Status

Pending — plan recorded 2026-06-27. No code changes made yet.

---

---

## 30. Multi-Database Context Support — Architecture & TODO List

> **Goal:** Allow the chatbot to serve multiple isolated database "contexts" — each with its own set of tables, schema files, vector collection, and system prompt — while keeping the underlying process flow (query → LLM → SQL → DB → response) fully intact. Some table names may be identical across contexts.

> **Decision:** Use **one PostgreSQL database with multiple PostgreSQL schemas** (one schema per context) within the **existing codebase**. See [Section 31](#31-database-structure-for-multi-context--decision--schema-design) for the full rationale, including why a separate codebase was rejected.

> **Implementation note (validated 2026-06-30):** Auth tables (`users`, `refresh_tokens`, `prompt_logs`) stay in the `public` schema — zero changes to `auth/models.py` or existing migrations. Only new tables (`db_contexts`, `user_contexts`) are created in the `app` schema. All work happens in a dedicated feature branch. See feature branch guidance below.

---

### Known Issues & Fixes (discovered during plan review, 2026-06-30)

| Priority | Issue | Location | Fix |
|---|---|---|---|
| P0 | `SET LOCAL search_path` fails silently with `autocommit=True` — queries run against wrong schema | `execute_query.py:37–38` / TODO 15 | Use explicit transaction (`conn.begin()`), include `public` in path — see TODO 15 |
| P1 | `ctx.schema_name` interpolated into SQL string with no validation — SQL injection risk | TODO 15 | Validate schema name against `^[a-z_][a-z0-9_]*$` before use |
| P1 | `ingest_vector_schema.py:21` hardcodes `"db_schema_vectors"` — not covered by TODO 7 | `application/ingest_vector_schema.py` | Add to TODO 7 touch-point list |
| P1 | Dual source of truth: config.yaml vs. DB for contexts — no reconciliation | TODOs 1, 3, 5 | config.yaml is the source of truth; DB is seeded from it at startup — see TODO 5 |
| P1 | `alembic/env.py` `include_object` will not detect `app`-schema ORM models for autogenerate | `alembic/env.py:26–35` | Fix before any `--autogenerate` run; write first migration by hand |
| P2 | `GRANT ALL PRIVILEGES ON ALL TABLES` does not cover future tables | Section 31 Bootstrap SQL | Add `ALTER DEFAULT PRIVILEGES` — see Section 31 |
| P2 | `user_contexts.is_default` has no DB-level uniqueness constraint | TODO 3 | Add partial unique index `WHERE is_default = TRUE` |
| P2 | No error when user has zero assigned contexts — unhandled 500 | TODO 18 | Return `HTTP 422` with clear message |
| P2 | Full schema docs injected on every request including greetings — high token cost | TODO 20 | Use `schema_summary.json` instead of full table docs for prompt injection |
| P3 | Context CRUD placed in `auth/repository.py` — wrong module | TODO 4 | Move to new `context/repository.py` |
| P3 | Context switching within same `thread_id` is undefined | TODO 10 | Document behavior — require new `thread_id` per context switch |
| P3 | `"No code changes needed for a new context"` claim is misleading | Section 31 | Corrected — lists all 9 required infrastructure steps |

---

### Feature Branch & Pre-flight

#### Create a dedicated feature branch

All work in Phases 1–8 must be done in a new branch, never directly on `develop`. The existing auth system is working and any regression there is a user-facing security failure.

```bash
git checkout develop
git pull origin develop
git checkout -b 30-feat-multi-context-schema-routing
```

#### `auth/models.py` — no changes needed ✅

The existing ORM models (`User`, `RefreshToken`, `PromptLog`) have no `schema=` argument and FK strings are bare (`"users.id"`). **Leave them exactly as they are.** The auth tables stay in the `public` schema permanently. The `get_auth_db()` session used by all auth endpoints never goes through `execute_query.py`, so the `SET LOCAL search_path` the SQL agent uses does not affect the auth system at all.

#### Existing migration files — do not edit ✅

The 4 existing migrations (`afab544e8c58`, `65c0b5446016`, `9927490b88de`, `1ab190c811b5`) are already applied. Do not edit them. Do not run `ALTER TABLE ... SET SCHEMA`. The first new migration is purely additive — see Section 31 for the correct migration template.

#### `alembic/env.py` — one fix required before autogenerate

The `include_object` function (lines 26–35) uses `name in Base.metadata.tables` to detect tables. For schema-qualified ORM models (e.g., `app.db_contexts`), the metadata key is `"app.db_contexts"` but `name` is just `"db_contexts"` — so autogenerate silently ignores new `app`-schema tables. **Write the first multi-context migration by hand** (not `--autogenerate`). Fix `include_object` on this branch before any future `--autogenerate` runs.

---

### Phase 1 — Context Registry (Config & DB Layer)

#### TODO 1 — Define a `DbContext` config schema in `config.yaml`

Add a `contexts:` block. Each entry contains: `context_id` (unique slug), `schema_name` (PostgreSQL schema, e.g. `context_sales`), `include_tables` (list), `system_prompt_path`, `router_prompt_path`, and `schema_dir` (e.g. `temp/context_sales/`). The `vector_collection_name` field maps to the `collection_name` used in `langchain_pg_collection`.

```yaml
contexts:
  sales:
    schema_name: context_sales
    include_tables: [orders, customer, product, inventory, v_sales_summary]
    system_prompt_path: src/ai_agentic_chatbot/prompts/context_sales/system_prompt.md
    router_prompt_path: src/ai_agentic_chatbot/prompts/context_sales/router_prompt.md
    schema_dir: temp/context_sales/
    vector_collection_name: context_sales_vectors
  hr:
    schema_name: context_hr
    include_tables: [orders, employee, department]
    system_prompt_path: src/ai_agentic_chatbot/prompts/context_hr/system_prompt.md
    router_prompt_path: src/ai_agentic_chatbot/prompts/context_hr/router_prompt.md
    schema_dir: temp/context_hr/
    vector_collection_name: context_hr_vectors
```

#### TODO 2 — Create a `DbContextRegistry` settings class

Create `infrastructure/context/context_settings.py`. Reads and validates the `contexts:` block from `config.yaml`. Exposes `get_context(context_id: str) -> DbContextConfig` (raises `KeyError` on unknown IDs). Follow the same singleton pattern as `DataSourceSettings`.

#### TODO 3 — Add `db_contexts` and `user_contexts` tables via Alembic

**`db_contexts` table** (schema `app`):

| Column | Type | Notes |
|---|---|---|
| `id` | BigInteger PK | autoincrement |
| `context_id` | String(100) | unique slug, e.g. `sales` |
| `display_name` | String(255) | human-readable label |
| `schema_name` | String(100) | PostgreSQL schema name, e.g. `context_sales` |
| `system_prompt_path` | Text | file path for this context's system prompt |
| `router_prompt_path` | Text | file path for router prompt |
| `schema_dir` | Text | local directory for temp schema files |
| `vector_collection_name` | String(255) | pgvector collection name |
| `include_tables` | JSONB | list of table names to introspect |
| `is_active` | Boolean | default True |
| `created_at` | DateTime | server_default now() |

**`user_contexts` table** (schema `app`, many-to-many: users ↔ contexts):

| Column | Type | Notes |
|---|---|---|
| `user_id` | BigInteger | FK → `public.users.id` ON DELETE CASCADE — auth tables stay in `public` |
| `context_id` | BigInteger | FK → `app.db_contexts.id` ON DELETE CASCADE |
| `is_default` | Boolean | one default context per user |

**Required partial unique index** — prevents two default contexts per user at the database level:

```sql
CREATE UNIQUE INDEX uq_user_default_context
ON app.user_contexts (user_id)
WHERE is_default = TRUE;
```

Add this index to the Alembic migration using `op.create_index()` with `postgresql_where=text("is_default = TRUE")`. Do not rely solely on application logic to enforce this constraint.

Run `alembic revision --autogenerate -m "add db_contexts and user_contexts"` after Phase 0 completes.

#### TODO 4 — Add context CRUD to a new `context/repository.py`

**Do NOT add to `auth/repository.py`** — that module already owns JWT tokens, user management, refresh token lifecycle, and prompt logging. Context management is a separate domain.

Create `src/ai_agentic_chatbot/context/repository.py`:

| Function | Purpose |
|---|---|
| `get_contexts_for_user(db, user_id)` | Returns all contexts the user is assigned to |
| `get_default_context(db, user_id)` | Returns the user's default context row, or `None` |
| `get_context_by_slug(db, context_id)` | Looks up a `db_contexts` row by its string slug |
| `assign_context_to_user(db, user_id, context_db_id, is_default)` | Inserts a `user_contexts` row; if `is_default=True`, clears the existing default first |
| `seed_contexts_from_config(db, registry)` | Called at startup — upserts config.yaml entries into `db_contexts` so the DB is always in sync with config |

#### TODO 5 — Add admin API endpoints for context management; resolve the source-of-truth conflict ✅ COMPLETED

**Source-of-truth decision (was missing from original plan):**

config.yaml is the **source of truth** for context definitions. The `db_contexts` table is a **DB mirror** populated by `seed_contexts_from_config()` at startup and used for user assignments. `POST /context/admin` must NOT be used to create new contexts — it only manages user assignments. To add a new context: add it to config.yaml first, then restart the service (which seeds it into the DB), then assign it to users.

**Amendment during implementation — router placement (2026-07-01):** the original plan said `File: auth/router.py`, reusing the existing superuser-gated router (`/auth/users`, etc.). That was implemented first, then reverted after review: it made `auth/router.py` import from `context.repository`/`context.schemas`, i.e. **auth depending on context** — the reverse of the dependency direction TODO 4 already established for the repository layer (context must depend on auth, never the other way). It also produced `/auth/admin/contexts` and `/auth/contexts` instead of the clean paths below.

**Final implementation:** a dedicated `context/router.py`, mounted in `server.py` via `app.include_router(context_router)` (the same one-line pattern already used for `auth_router` — unavoidable, since FastAPI has no router auto-discovery). It imports `get_current_user` / `get_auth_db` from `auth.dependencies` and `get_user_by_username` from `auth.repository` — a one-way dependency on `auth`. `auth/router.py` has zero knowledge of `context`.

File: `context/router.py` (prefix `/context`)

| Endpoint | Method | Notes |
|---|---|---|
| `/context/admin` | GET | List all active contexts from DB (superuser only) |
| `/context/admin/{context_slug}/users/{username}` | PUT | Assign context to user (superuser only); `is_default` query param clears any prior default first |
| `/context/admin/{context_slug}/users/{username}` | DELETE | Remove user's context access (superuser only); HTTP 404 if not assigned |
| `/context` | GET | Return calling user's allowed contexts with an `is_default` flag (for UI switcher) — this also satisfies TODO 19, which specified the same endpoint; no separate work was needed there |

> Removing `POST /context/admin` from the original plan — context creation is controlled by config.yaml, not the API.

**Repository additions beyond TODO 4's original list:** `context/repository.py` needed two more functions to back these endpoints — `list_active_contexts(db)` (all active `db_contexts` rows, for the admin list) and `remove_context_assignment(db, user_id, context_db_id) -> bool` (for the DELETE endpoint, returns `False` if no assignment existed so the route can 404).

**Schema gap fixed along the way:** `db_contexts.display_name` is `NOT NULL` (TODO 3), but neither `DbContextConfig` (TODO 2) nor `config.yaml`'s `contexts:` block defined one. Added `display_name: Optional[str] = None` to `DbContextConfig` (defaults to the context slug if omitted) so `seed_contexts_from_config` has a value to write.

**Verified against the live `ai_chatbot_db_v2` database** using real `admin`/`dcc` accounts via `TestClient` (with lifespan startup so `DataSourceFactory` is initialized): superuser list, non-superuser 403, assign with `is_default=true`, self-service list reflecting the assignment, unassign (204), repeat unassign (404), and confirmation the list reverts to empty. No test data left behind.

**Unrelated bug found and fixed during verification:** `users_id_seq` was desynced from `public.users` (`last_value=2` while `id=3` already existed for user `dcc`), which would have broken the next `/auth/register` call with a `UniqueViolation`. Fixed via `SELECT setval('users_id_seq', (SELECT MAX(id) FROM users))`.

---

### Phase 2 — Schema Pipeline Namespacing

#### TODO 6 — Namespace all temp/schema files under `temp/{context_id}/`

| File | Change |
|---|---|
| `schema_extractor/SaveSchemaJson.py:13` | Accept `schema_dir: Path` parameter; write to `schema_dir / "db_schema.json"` instead of hardcoded `Path.cwd() / "temp" / "db_schema.json"` |
| `application/transform_schema_to_text.py:19–21, 80–102` | Replace hardcoded `temp/` paths with `schema_dir` argument passed through from the caller |
| `application/ingest_vector_schema.py:9` | Accept and pass `schema_dir` to `VectorSchemaBuilder` and `SaveSchemaJson` |

#### TODO 7 — Namespace the pgvector collection per context

**Three files, not two** — the original plan missed `ingest_vector_schema.py`:

| File | Change |
|---|---|
| `infrastructure/vector_store/pgvector_store.py:26` | Replace hardcoded `"db_schema_vectors"` with a `collection_name` constructor argument; remove the module-level `_vector_store` singleton (lines 106–114) and `get_vector_store()` function entirely |
| `application/ingest_vector_schema.py:21` | Replace hardcoded `"db_schema_vectors"` — pass `collection_name=ctx.vector_collection_name` when constructing `PgVectorSchemaStore` |
| `schema_extractor/vector_schema_builder.py:76` | Change UUID seed from `uuid5(NAMESPACE_DNS, table_name)` to `uuid5(NAMESPACE_DNS, f"{context_id}:{table_name}")` to prevent cross-context ID collisions |
| `infrastructure/vector_store/pgvector_store.py:30–34` | Pass `schema="vector_store"` to `PGVector(...)` constructor so all pgvector tables land in the `vector_store` PostgreSQL schema |

Also check `server.py` for any remaining direct calls to the old `get_vector_store()` singleton (e.g., `reset_collection()` in the `/ingest` endpoint) and update them to use the context-parameterized constructor.

#### TODO 8 — Remove the module-level `SchemaLoader` singleton

File: `schema_extractor/schema_loader.py:298–307`

- Eliminate the `_schema_loader` global (the self-identified TODO comment at line 298 flags this exact issue).
- Replace `get_schema_loader()` with a factory function `get_schema_loader(context_id: str) -> SchemaLoader` that accepts a `schema_dir` and `context_id`, and caches instances in a `Dict[str, SchemaLoader]` keyed by `context_id`.
- **Cache invalidation note:** `SchemaLoader` reads YAML files from disk on each method call — it has no internal in-memory cache of file contents. The per-context dict cache therefore serves only as a way to avoid re-constructing the object. If `SchemaLoader` ever gains internal file caching, this dict must add an invalidation path (e.g., evict and re-create the entry after `/schemaText` completes for that context). Document this dependency with a comment in `get_schema_loader()`.

#### TODO 9 — Update admin setup routes to accept `context_id`

File: `server.py` — routes `/schemaJson`, `/schemaText`, `/ingest`

- Add `context_id: str` as a query parameter on each route.
- Load the matching `DbContextConfig` from `DbContextRegistry`.
- Pass `include_tables`, `schema_dir`, `vector_collection_name`, `context_id`, and `schema_name` (PostgreSQL schema) to each pipeline step.
- Remove the hardcoded `include_tables` list at line 138.
- Pass `SchemaExtractionConfig(include_schemas=[ctx.schema_name], include_tables=ctx.include_tables)` to `SchemaExtractor` — the `include_schemas` field already exists in `SchemaExtractionConfig` but has never been populated.

---

### Phase 3 — Agent State & Prompt Propagation

#### TODO 10 — Add `db_context_id: str` to both state models

| File | Change |
|---|---|
| `agent/state.py` | Add `db_context_id: str` field to `AgentState` |
| `agent/subgraphs/sql_query/state.py` | Add `db_context_id: str` field to `SQLSubgraphState` |
| `agent/graph.py` | When initializing `SQLSubgraphState`, copy `db_context_id` from `AgentState` |

**Context switching within the same `thread_id`:** LangGraph's MemorySaver checkpoint stores `db_context_id` per `thread_id`. If a client sends two requests with different `context_id` values on the same `thread_id`, the second request's value will override the checkpoint. This is undefined and potentially dangerous behavior. **Document this constraint at the API level:** a client must use a new `thread_id` when switching contexts. Return `HTTP 400` if the incoming `context_id` differs from the context stored in the active checkpoint for that `thread_id`.

#### TODO 11 — Make `get_system_prompt()` context-aware

File: `utils/prompt_loader.py:35–42`

- Add optional `system_prompt_path: str = None` parameter.
- When provided, read from that path directly. When `None`, fall back to the `SYSTEM_PROMPT_PATH` env var (preserving backward compatibility for the single-context case during transition).
- Same change for `get_router_prompt()`.

#### TODO 12 — Update the router node to use context-specific prompt and schema

File: `agent/router.py:73–87`

- Read `db_context_id` from state.
- Call `DbContextRegistry.get_context(db_context_id)` to get the config.
- Call context-aware `get_system_prompt(ctx.router_prompt_path)` and `get_schema_loader(db_context_id).load_schema_summary()`.
- Note: line 75 currently instantiates `SchemaLoader()` directly instead of using a registry call. Replace with `get_schema_loader(db_context_id)`.

---

### Phase 4 — SQL Subgraph Nodes

#### TODO 13 — Make `retrieve_schemas` context-aware

File: `agent/subgraphs/sql_query/nodes/retrieve_schemas.py:15–55`

- Read `db_context_id` from state.
- Call `get_schema_loader(db_context_id)` for the DDL map.
- Instantiate `PgVectorSchemaStore(collection_name=ctx.vector_collection_name)` instead of the removed singleton.

#### TODO 14 — Make `generate_sql` context-aware

File: `agent/subgraphs/sql_query/nodes/generate_sql.py:55`

- Engine stays `get_engine("postgresql.primary")` (same database).
- Pass context-specific system prompt to the LLM call via context-aware `get_system_prompt(ctx.system_prompt_path)`.
- For glossary/column hints: pass `schema_name=ctx.schema_name` to the hint-fetching functions (see TODO 16).

#### TODO 15 — Make `execute_query` context-aware — `SET LOCAL search_path` ⚠️ CRITICAL FIX

File: `agent/subgraphs/sql_query/nodes/execute_query.py:37–38`

The original plan's code is **wrong and will silently query the wrong schema:**

```python
# WRONG — do not use this
with engine.connect() as conn:
    conn.execute(text(f"SET LOCAL search_path = {ctx.schema_name}, app"))
    result = conn.execute(text(sql_query).execution_options(autocommit=True))
```

**Why it fails:** `SET LOCAL` is transaction-scoped. With `autocommit=True` on the second execute, SQLAlchemy commits the implicit transaction before running the SQL query. The commit ends the transaction, clearing `SET LOCAL`. The SQL query then runs in a new transaction against the default `search_path` (typically `public`) — not `context_hr`. No error is raised; data from the wrong schema is silently returned.

**Correct implementation:**

```python
# CORRECT — explicit transaction via conn.begin()
import re

ctx = DbContextRegistry.get_context(state["db_context_id"])

# Validate schema_name before string interpolation (SET cannot use bound params)
if not re.match(r'^[a-z_][a-z0-9_]*$', ctx.schema_name):
    raise ValueError(f"Invalid schema name: {ctx.schema_name!r}")

with engine.connect() as conn:
    with conn.begin():                                # explicit transaction block
        conn.execute(
            text(f"SET LOCAL search_path TO {ctx.schema_name}, app, public")
        )
        # "public" fallback covers auth tables (users, refresh_tokens, prompt_logs)
        # which stay in the public schema permanently
        result = conn.execute(text(sql_query))        # same transaction — SET LOCAL is active
        rows = result.fetchmany(MAX_QUERY_RESULTS)
# transaction commits on conn.begin() exit; SET LOCAL resets automatically — no pool leak
```

The `with conn.begin()` block opens an explicit transaction. Both statements execute within it. `SET LOCAL` is strictly scoped to that transaction and resets automatically when it commits — the connection returned to the pool is clean with no lingering `search_path` side-effects.

**Also remove `autocommit=True`** from the existing `execute_query.py` — it is incompatible with `SET LOCAL` and is deprecated in SQLAlchemy 2.x for this pattern anyway.

#### TODO 16 — Make glossary/column hint lookups context-aware ✅ COMPLETED (during TODO 14)

**The doc's original plan here was wrong.** `business_glossary` and `schema_metadata` do live in each context's PostgreSQL schema (confirmed: `demo_01.business_glossary`, `demo_01.schema_metadata`), so the assumption was "rely on the `SET LOCAL search_path` set in TODO 15 — no additional changes needed." But `fetch_glossary_hints`/`fetch_column_hints` open their **own separate `engine.connect()`** calls in `glossary_lookup.py` — a different connection than the one `execute_query.py` sets `SET LOCAL search_path` on in TODO 15. `SET LOCAL` is transaction/connection-scoped, so it has zero effect here; relying on it would have left these lookups silently broken (always querying `public`, finding nothing) for any non-`public`-schema context.

**Actual fix (done as part of TODO 14, since the two changes couldn't be separated without breaking the app):** both functions in `glossary_lookup.py` now take a required `schema_name: str` parameter and schema-qualify their queries directly (`FROM {schema_name}.business_glossary`, `FROM {schema_name}.schema_metadata`), validated against `^[a-z_][a-z0-9_]*$` before interpolation — same identifier-validation pattern as TODO 15, since `SET`/table names can't use bound params. `generate_sql.py` resolves `ctx.schema_name` via the context registry and passes it through at both call sites.

Verified live: confirmed both tables exist in `demo_01` (not `public`), inserted a throwaway glossary row, confirmed `fetch_glossary_hints` correctly found it via the schema-qualified query, then cleaned it up. Also confirmed the schema-name validator blocks an injection attempt (`"demo_01; DROP TABLE users; --"`).

---

### Phase 5 — API Layer

#### TODO 17 — Add `context_id` to `StreamRequest`

File: `agent/schema.py`

```python
class StreamRequest(BaseModel):
    thread_id:  str
    messages:   list[Message]
    context_id: Optional[str] = None   # if None, use user's default context
```

#### TODO 18 — Update the `/stream` route to resolve and inject context

File: `server.py:245`

After authenticating the user (JWT), resolve `context_id` in this order:

1. If `stream_request.context_id` is provided:
   - Verify the user is in `user_contexts` for that context → raise `HTTP 403` if not.
   - Verify the `context_id` matches the checkpoint for this `thread_id` (if an existing checkpoint exists) → raise `HTTP 400` with message `"Context switch requires a new thread_id"` if they differ.
2. If `None`, call `get_default_context(db, current_user.id)`.
   - If that also returns `None` (user has no assigned contexts): raise `HTTP 422` with message `"No context assigned to this user. Contact an administrator."` — do NOT allow this to propagate to a 500 error.
3. Inject `db_context_id` into the initial `AgentState` before calling `graph.astream_events()`.

#### TODO 19 — Add `GET /contexts` endpoint for the frontend ✅ COMPLETED (as part of TODO 5)

Implemented as `GET /context` in `context/router.py` — see TODO 5 for the full router placement decision. Returns the list of contexts the current user is allowed to access, with `context_id`, `display_name`, and `is_default` flag, used by the frontend to render a context-switcher UI element. No separate implementation was needed once TODO 5 was done.

---

### Phase 6 — System Prompt Refactor

#### TODO 20 — Decouple table knowledge from `system_prompt.md`

The current `system_prompt.md` has 5 table names, column listings, and ~50 hardcoded SQL examples baked in statically. This makes it unusable for a second context.

**Token cost warning:** The original plan proposed injecting `SchemaLoader.get_table_docs_for_search()` as `{schema_context}` into the system prompt on every request. That method returns full LLM-enriched documentation for every table — potentially several thousand tokens per request, including greetings, routing turns, and clarification turns. Use `SchemaLoader.load_schema_summary()` instead — it loads `schema_summary.json`, which is a compact table-name and purpose summary suitable for routing context. Reserve full schema docs for the SQL generation node only (which already receives DDLs via `retrieved_tables`).

**Changes:**

- Remove the static table/column knowledge from `system_prompt.md` and replace it with a `{schema_summary}` placeholder (not `{schema_context}` — use the summary, not the full docs).
- Inject `ctx_loader.load_schema_summary()` at the `{schema_summary}` placeholder in `prompt_loader.py`.
- Create a separate `system_prompt.md` per context under `prompts/context_{id}/system_prompt.md`, keeping only context-independent instructions in a shared base template.

---

### Phase 7 — Pre-launch Dead Code Cleanup (Optional but Recommended)

Do this before starting Phase 1 to reduce noise during review:

| Location | Dead Code | Action |
|---|---|---|
| `agent/graph.py:~150–430` | `format_sql_response`, `format_sql_response_with_visualization`, `format_as_markdown_table`, `_analyze_for_visualizations`, `_format_data_values`, `_generate_data_summary` — never called | Delete |
| `agent/subgraphs/sql_query/nodes/retrieve_schemas.py:58–186` | Commented-out `_semantic_search` function | Delete |
| `schema_extractor/schema_loader.py:231–261` | Unused `_generate_ddl_from_doc` method | Delete |

---

### Phase 8 — Testing & Hardening

#### TODO 21 — Write integration tests for context isolation

- Verify a query from context A never touches context B's tables or vector collection.
- Verify a user not assigned to any context gets `HTTP 422`, not `HTTP 500`.
- Verify a user not assigned to a specific context gets `HTTP 403` when requesting that context.
- Verify `SET LOCAL search_path` resets correctly between requests (pool connection reuse test) — connect to the same pooled connection twice and confirm the second query does not inherit the first request's search_path.
- Verify context switching on the same `thread_id` returns `HTTP 400`.

#### TODO 22 — Seed a second context end-to-end

1. Add the new context to `config.yaml` under `contexts:`.
2. Create the PostgreSQL schema: `CREATE SCHEMA context_hr;`
3. Grant permissions (see Section 31 Bootstrap SQL).
4. Restart the service so `seed_contexts_from_config()` registers the new context in `db_contexts`.
5. Call `GET /schemaJson?context_id=hr` → `GET /schemaText?context_id=hr` → `GET /ingest?context_id=hr`.
6. Assign the context to a test user via `PUT /admin/contexts/hr/users/{username}`.
7. Run a query through `POST /stream` with `context_id=hr` and verify the SQL targets `context_hr.` tables.
8. Confirm a user without the `hr` assignment gets `HTTP 403` when requesting `context_id=hr`.

---

### Full Touch-Point Summary

| Layer | Files to Change | Nature of Change |
|---|---|---|
| **Pre-flight** | `alembic/env.py` | Fix `include_object` for `app`-schema autogenerate (before first `--autogenerate` run) |
| Config | `config.yaml` | Add `contexts:` block |
| DB models + migrations | New Alembic migration | `db_contexts`, `user_contexts` tables + partial unique index |
| Context registry | New `infrastructure/context/context_settings.py` | `DbContextRegistry` singleton |
| Context CRUD | New `context/repository.py` | Context-user assignment functions (not in auth/repository.py) |
| Schema pipeline | `SaveSchemaJson.py`, `transform_schema_to_text.py`, `ingest_vector_schema.py` | Accept `schema_dir`, `context_id` params |
| Vector store | `pgvector_store.py`, `ingest_vector_schema.py`, `vector_schema_builder.py` | Remove singleton; parameterize collection name; namespace UUIDs; add `schema="vector_store"` |
| Schema loader | `schema_loader.py` | Remove global singleton; add per-context cache with invalidation note |
| Agent state | `agent/state.py`, `sql_query/state.py` | Add `db_context_id` field |
| Prompt loader | `prompt_loader.py`, `system_prompt.md` | Context-aware path resolution; add `{schema_summary}` placeholder (not full docs) |
| Router node | `agent/router.py` | Use context-specific prompt + per-context `get_schema_loader()` call |
| SQL nodes | `retrieve_schemas.py`, `generate_sql.py`, `execute_query.py`, `glossary_lookup.py` | Context-aware schema loader, vector store, `SET LOCAL` with `engine.begin()` |
| API | `agent/schema.py`, `server.py` | Add `context_id` to `StreamRequest`; resolve + inject at `/stream`; handle zero-context user with 422 |
| Admin routes | `context/router.py` (new, prefix `/context`) | Context-user assignment endpoints (no POST /context/admin); depends one-way on `auth.dependencies`/`auth.repository`, never the reverse |
| System prompts | `prompts/context_{id}/system_prompt.md` | One prompt file per context |

> **Key insight:** The `DataSourceFactory` infrastructure already supports multiple named datasources. Since all contexts share one PostgreSQL database, `get_engine("postgresql.primary")` remains unchanged in all nodes — only the `SET LOCAL search_path` injection in `execute_query.py` (using `engine.begin()`) changes how the engine resolves table names at runtime.

---

## 31. Database Structure for Multi-Context — Decision & Schema Design

> **Question:** Should contexts be isolated using (A) multiple PostgreSQL schemas within one database, or (B) separate databases per context? And should this be done in the **existing codebase** or a **new separate codebase**?

### Why a Separate Codebase Was Rejected

Before comparing schema options, a separate codebase was evaluated and rejected for one decisive reason: **the auth system is not separable without unacceptable cost.**

Every user account, JWT session, refresh token, daily prompt limit, and audit log lives in tables managed by this codebase. A separate codebase for a new context would either:

- **Duplicate the entire auth system** → two disconnected user databases, two logins, doubled security surface area, and two refresh token revocation trees to maintain independently.
- **Depend on this codebase's database externally** → both services would share the same JWT secret and `users` table anyway, making the separation meaningless — just a worse version of the same architecture with an added network hop and two deployments to keep in sync.

For 2–5 internal contexts in one organization, the refactor described in Section 30 is the correct path.

---

### Constraints that drove the schema decision

| Constraint | Value |
|---|---|
| Expected number of contexts | 2–5 (small, fixed set) |
| Ownership | All internal domains within the same organization |
| Cross-context queries needed | Yes — some admin/reporting queries join data across contexts |

### Option Comparison

| Concern | Option A — Multiple PostgreSQL Schemas | Option B — Separate Databases |
|---|---|---|
| Table name collisions | Handled natively — `context_a.orders` and `context_b.orders` coexist | Handled — each database has its own `public.orders` |
| Connection pool usage | 15 connections total regardless of context count | 15 connections × N databases (15 × 5 = 75 max) |
| Cross-context queries | Native cross-schema JOIN: `SELECT * FROM context_sales.orders JOIN context_hr.employee` | Requires PostgreSQL foreign data wrappers (FDW) or ETL — complex and slow |
| Operational overhead | One database to manage, back up, monitor | N databases — N connection strings, N Alembic runs, N backup jobs |
| Security isolation | Schema-level (PostgreSQL role grants per schema). Note: `chatbot_user` has grants on all schemas — a `SET LOCAL` bug would cause silent cross-schema access (see TODO 15 fix) | Database-level (true hard isolation) |
| LLM SQL generation impact | Zero — `SET LOCAL search_path TO context_X, app, public` means LLM writes plain `SELECT * FROM orders`, PostgreSQL resolves to the right schema | Zero — engine points to the right database |
| Existing code changes | `execute_query.py:37–38`: add `conn.begin()` + `SET LOCAL`; `auth/models.py` unchanged; engine unchanged | 4 files need dynamic engine resolution; N connection strings to manage |
| Alembic migrations | New migration per context using `op.create_table(..., schema="context_<slug>")` | Re-run Alembic with different `POSTGRESQL_DB` env var per database |
| pgvector (`langchain_postgres`) | Pass `schema="vector_store"` to `PGVector(...)` constructor | Separate `langchain_pg_collection` table in each database |

**Security caveat on Option A:** Schema-level isolation is not equivalent to database-level isolation. The `chatbot_user` application role has `USAGE` grants on all schemas. If the LLM ever generates a schema-qualified query (e.g., `SELECT * FROM context_hr.employee`) during a `context_sales` session — through a misconfiguration or schema doc contamination in the vector store — the query will succeed. The `SET LOCAL search_path` mechanism is a convention enforced by application code, not a hard database boundary. For the stated use case (2–5 internal domains, same organization), this risk is acceptable. For multi-tenant or multi-organization deployments, Option B (separate databases) would be required.

### Decision: Option A — One Database, Multiple PostgreSQL Schemas

**Reasons:**
1. Cross-context admin queries work natively with schema-qualified JOINs.
2. Single connection pool (15 connections) — no resource explosion.
3. Single database to manage, back up, and monitor.
4. `SET LOCAL search_path TO context_X, app, public` (via `conn.begin()`) is the only change to the SQL execution path — the LLM writes unqualified table names, PostgreSQL resolves them against the context schema first.
5. `SchemaExtractor` already accepts `schema=` in `inspector.get_table_names(schema=schema_name)` calls — the extraction layer is already schema-aware.

### Recommended Schema Layout

```
your_database (ai_chatbot_db)
│
├── public                     ← auth tables stay here permanently — zero migration needed
│   ├── users                  ← auth/models.py unchanged; no __table_args__ added
│   ├── refresh_tokens
│   ├── prompt_logs
│   └── alembic_version
│
├── app                        ← new metadata tables only (created by new migration)
│   ├── db_contexts
│   └── user_contexts          ← FK user_id → public.users.id (cross-schema FK, fully supported)
│
├── context_sales              ← first business context
│   ├── orders
│   ├── customer
│   ├── product
│   ├── inventory
│   └── v_sales_summary
│
├── context_hr                 ← second business context
│   ├── orders                 ← same name as context_sales.orders — no collision
│   ├── employee
│   └── department
│
└── vector_store               ← all langchain pgvector tables, one per context
    ├── langchain_pg_collection   (collection_name = "context_sales_vectors", etc.)
    └── langchain_pg_embedding
```

### How `SET LOCAL search_path` Works at Runtime (Corrected)

```
User sends POST /stream with context_id = "hr"
         │
         ▼
/stream resolves DbContextConfig → schema_name = "context_hr"
         │
         ▼
agent state: db_context_id = "hr"
         │
         ▼
execute_query node:
    # Validate schema_name (SQL injection guard — SET cannot use bound params)
    if not re.match(r'^[a-z_][a-z0-9_]*$', ctx.schema_name):
        raise ValueError(...)

    with engine.connect() as conn:
        with conn.begin():                          # ← explicit transaction block
            conn.execute(text("SET LOCAL search_path TO context_hr, app, public"))
            #  search_path priority: context_hr → app → public
            #  "public" covers auth tables (users etc.) that stay in public schema
            result = conn.execute(text("SELECT * FROM orders WHERE ..."))
            #                                          ^^^^^^
            #  LLM wrote this — no schema prefix needed
            #  PostgreSQL resolves: context_hr.orders
        # conn.begin() exits → transaction commits → SET LOCAL resets
         │
         ▼
Connection returns to pool — search_path is back to database default (clean)
(SET LOCAL is strictly transaction-scoped — zero pool leak risk)
```

> **Why NOT `autocommit=True`:** `SET LOCAL` is transaction-scoped. With `autocommit=True`, each statement runs in its own implicit transaction that immediately commits. The `SET LOCAL` commits and its scope ends before the SQL query even starts, so the query runs against the database-default `search_path` (typically `public` only). No error is raised — wrong schema data is silently returned. The explicit `conn.begin()` block keeps both statements in one transaction.

### Auth ORM Models — No Changes Required ✅

`auth/models.py` does not need any changes. All three ORM classes (`User`, `RefreshToken`, `PromptLog`) stay exactly as they are — no `__table_args__`, no `schema=` argument, no FK string changes. The auth tables remain in `public` permanently.

The auth session (`get_auth_db()`) is completely isolated from the SQL agent's `execute_query.py` connection. The `SET LOCAL search_path` the agent applies never affects auth queries.

### Required Alembic Changes — Purely Additive New Migration

**DO NOT edit the 4 existing migration files.** They have already been applied; editing them does nothing, and retroactively altering them can corrupt downgrade paths.

**The new migration is purely additive — only CREATE statements, no ALTER or DROP on existing tables:**

```python
# alembic/versions/XXXX_add_app_schema_and_context_tables.py
# Write this manually — do NOT use --autogenerate (env.py include_object bug)

def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS app")
    op.execute("CREATE SCHEMA IF NOT EXISTS vector_store")
    op.create_table(
        "db_contexts",
        # ... columns (see TODO 3 in Section 30)
        schema="app",
    )
    op.create_table(
        "user_contexts",
        # user_id FK references public.users — cross-schema FK is fully supported in PostgreSQL
        # ForeignKeyConstraint(['user_id'], ['public.users.id'], ondelete='CASCADE')
        schema="app",
    )
    op.create_index(
        "uq_user_default_context",
        "user_contexts",
        ["user_id"],
        schema="app",
        unique=True,
        postgresql_where=text("is_default = TRUE"),
    )

def downgrade() -> None:
    op.drop_index("uq_user_default_context", table_name="user_contexts", schema="app")
    op.drop_table("user_contexts", schema="app")
    op.drop_table("db_contexts", schema="app")
    op.execute("DROP SCHEMA IF EXISTS vector_store")
    op.execute("DROP SCHEMA IF EXISTS app")
```

**Safe downgrade check:** `alembic downgrade -1` drops only the tables and schemas this migration created. It does not touch `public.users`, `public.refresh_tokens`, or `public.prompt_logs`. The auth system remains fully functional after a downgrade.

New context migrations (one per context) use `schema="context_<slug>"` in each `op.create_table()` — same purely additive pattern.

### PostgreSQL Bootstrap (run once per environment)

Auth tables (`users`, `refresh_tokens`, `prompt_logs`) stay in `public` — no grants needed for them, they already have the right permissions. Only grant new schemas:

```sql
-- Create new schemas (public already exists — do not recreate it)
CREATE SCHEMA IF NOT EXISTS app;
CREATE SCHEMA IF NOT EXISTS context_sales;
CREATE SCHEMA IF NOT EXISTS context_hr;
CREATE SCHEMA IF NOT EXISTS vector_store;

-- Grant usage on new schemas
GRANT USAGE ON SCHEMA app TO chatbot_user;
GRANT USAGE ON SCHEMA context_sales TO chatbot_user;
GRANT USAGE ON SCHEMA context_hr TO chatbot_user;
GRANT USAGE ON SCHEMA vector_store TO chatbot_user;

-- Cover tables created by FUTURE Alembic migrations in the new schemas
-- (GRANT ALL ON ALL TABLES only covers tables that exist at this moment;
--  ALTER DEFAULT PRIVILEGES covers tables created later by migrations)
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON TABLES TO chatbot_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON SEQUENCES TO chatbot_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA context_sales GRANT ALL ON TABLES TO chatbot_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA context_hr GRANT ALL ON TABLES TO chatbot_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA vector_store GRANT ALL ON TABLES TO chatbot_user;
```

### Adding a Third Context — Complete Steps

The following steps are all required. "No code changes" is accurate for Python application code only — there are 6+ infrastructure and file-system steps:

1. `CREATE SCHEMA context_finance;` in PostgreSQL.
2. Add `ALTER DEFAULT PRIVILEGES IN SCHEMA context_finance GRANT ALL ON TABLES TO chatbot_user;`.
3. Add the context entry to `config.yaml` under `contexts:`.
4. Create `prompts/context_finance/system_prompt.md` and `router_prompt.md`.
5. Create `temp/context_finance/` directory.
6. Restart the service — `seed_contexts_from_config()` registers the new entry in `db_contexts`.
7. Populate the business data tables in `context_finance` schema (out of scope for this app).
8. Call `GET /schemaJson?context_id=finance` → `GET /schemaText?context_id=finance` → `GET /ingest?context_id=finance`.
9. Assign the context to users via `PUT /admin/contexts/finance/users/{username}`.

---

*Generated: 2026-05-29 | Updated: 2026-06-30 | Repository: `ai-agentic-chatbot` | Branch: `develop`*
