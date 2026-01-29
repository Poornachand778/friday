# 🎬 Friday AI Knowledge Transfer Document 2

## Database & Screenplay Management System Implementation

**Document Version:** 2.0  
**Created:** 28 September 2025  
**Focus:** Database Schema Extensions, Screenplay Management, MCP Integration  
**Previous:** See `FRIDAY_AI_KT_DOCUMENT.md` for Iteration 1 foundation

---

## 📋 **Executive Summary**

This document captures the major infrastructure advances made between Iteration 1 and Iteration 2, focusing on:

- **Database schema expansion** for screenplay management
- **Vector search capabilities** for multilingual scene retrieval
- **MCP server foundation** for agent tool integration
- **Scene versioning and ordering** system for script development

**Status:** Database schema implemented ✅ | MCP server scaffolded ⚠️ | Ready for Iteration 2 fine-tuning

---

## 🗄️ **Database Schema Evolution**

### **Core Training Tables (Unchanged from Iter1)**

- `datasets` - Training set registry
- `model_versions` - Adapter/base model tracking
- `training_runs` - Fine-tune execution logs
- `eval_suites`, `eval_cases`, `eval_runs`, `eval_results` - Evaluation framework
- `memory_entries` - Long-term memory placeholder
- `artifacts` - Model artifacts and reports

### **New Screenplay Management Tables**

#### **`script_projects`** - Film/Series Registry

```sql
CREATE TABLE script_projects (
    id INTEGER PRIMARY KEY,
    slug VARCHAR(128) UNIQUE NOT NULL,     -- "aa-janta-naduma"
    title VARCHAR(256) NOT NULL,           -- "Aa Janta Naduma"
    logline TEXT,                          -- Brief story hook
    status VARCHAR(64) DEFAULT 'draft',    -- draft/active/completed
    notes TEXT,                            -- Production notes
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### **`script_scenes`** - Individual Scene Management

```sql
CREATE TABLE script_scenes (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES script_projects(id),
    scene_code VARCHAR(32) NOT NULL,       -- "SCN001", "SCN032"
    title VARCHAR(256),                    -- "Proposal Scene"
    summary TEXT,                          -- Brief scene description
    tags JSON DEFAULT '[]',                -- ["comedy", "outdoor", "climax"]
    canonical_text TEXT NOT NULL,          -- Current scene content
    narrative_order FLOAT DEFAULT 0,       -- 1.0, 1.5, 2.0 (reordering)
    status VARCHAR(64) DEFAULT 'active',   -- active/backlog/alternate
    current_revision_id INTEGER REFERENCES script_revisions(id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(project_id, scene_code)
);
```

#### **`script_revisions`** - Version History

```sql
CREATE TABLE script_revisions (
    id INTEGER PRIMARY KEY,
    scene_id INTEGER REFERENCES script_scenes(id),
    revision_number INTEGER NOT NULL,      -- 1, 2, 3...
    author VARCHAR(128) DEFAULT 'user',    -- 'user' or 'friday'
    notes TEXT,                           -- "Added humor", "TTS feedback"
    content TEXT NOT NULL,                -- Full scene text snapshot
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(scene_id, revision_number)
);
```

#### **`scene_links`** - Scene Relationships

```sql
CREATE TABLE scene_links (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES script_projects(id),
    from_scene_id INTEGER REFERENCES script_scenes(id),
    to_scene_id INTEGER REFERENCES script_scenes(id),
    relation_type VARCHAR(64) NOT NULL,    -- "sequence", "flashback", "alternate"
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### **`script_scene_embeddings`** - Vector Search

```sql
CREATE TABLE script_scene_embeddings (
    id INTEGER PRIMARY KEY,
    scene_id INTEGER REFERENCES script_scenes(id),
    revision_id INTEGER REFERENCES script_revisions(id),
    model VARCHAR(128) NOT NULL,           -- "distiluse-base-multilingual-cased-v2"
    vector TEXT NOT NULL,                  -- JSON serialized embedding
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🎭 **Screenplay Management Workflow**

### **Data Import Process**

1. **Script Parsing**: `data/film/scripts/aa_janta_naduma_draft.md` → Database
2. **Scene Extraction**: 32 scenes identified (`SCN001`-`SCN032`)
3. **Initial Revisions**: Each scene gets revision #1 with current content
4. **Embedding Generation**: Multilingual vectors created for semantic search
5. **Sequence Links**: Auto-generated "sequence" relationships (SCN001→SCN002→...)

### **Scene Management Features**

#### **Flexible Ordering System**

- **`narrative_order`** (FLOAT): Allows easy resequencing without renumbering
  - Insert scene 3.5 between scenes 3 and 4
  - Move climax scene from position 30 to position 5
  - Maintain order integrity during discussions

#### **Status Tracking**

- **`active`**: Finalized scenes in current script
- **`backlog`**: Written but not placed in narrative
- **`alternate`**: Alternative versions for comparison
- **`library`**: Scenes for potential future projects

#### **Version Control**

- **Immutable History**: Every edit creates new revision
- **Author Tracking**: User vs Friday edits
- **Context Preservation**: Tomorrow's session continues where left off
- **Diff Capability**: Compare revisions for TTS/storyboard updates

---

## 🔍 **Vector Search & Multilingual Support**

### **Embedding Model Selection**

- **Model**: `sentence-transformers/distiluse-base-multilingual-cased-v2`
- **Rationale**: Fast, lightweight, supports Telugu Unicode + English
- **Performance**: Optimized for speed over perfection (1000+ scenes target)

### **Search Capabilities**

```python
# Example queries that work:
"proposal scene" → finds SCN015 (Telugu dialogue, English description)
"Niha confronts Arjun" → semantic match regardless of language mix
"క్లైమాక్స్ దృశ్యం" → Telugu query finds English-described climax
```

### **Implementation Details**

- **Storage**: Postgres `TEXT` column with JSON serialized vectors
- **Indexing**: Scene content + summary embedded per revision
- **Refresh**: `scripts/update_scene_embeddings.py --recompute` after edits
- **Fallback**: Direct scene_code/ID lookup for exact references

---

## 🔧 **MCP Server Integration Architecture**

### **Current Status: Dual Interface (FastAPI + MCP)**

- **HTTP service**: `mcp/scene_manager/service.py` (FastAPI) remains for REST clients.
- **MCP server**: `mcp/scene_manager/server.py` now exposes the same operations through a JSON-RPC (stdin/stdout) loop aligned with the Model Context Protocol tool contract.

### **Available MCP Tools**

| Tool | Description | Key Arguments |
| ---- | ----------- | ------------- |
| `scene_search` | Semantic retrieval across scenes | `query`, `top_k`, `project_slug?` |
| `scene_get` | Fetch canonical scene detail | `scene_code` or `scene_id`, `project_slug?` |
| `scene_update` | Create revision / update status/order | `scene_code`, `canonical_text?`, `narrative_order?`, `status?`, `notes?` |
| `scene_reorder` | Reposition scenes via before/after references | `scene_code`, `after_scene?`, `before_scene?`, `project_slug?` |
| `scene_link` | Persist scene relationships | `from_scene`, `to_scene`, `relation_type?`, `project_slug?` |

### **Running the MCP Server**

```bash
# Example: run with default project slug
python mcp/scene_manager/server.py --log-level INFO

# Send a request (newline-delimited JSON) from another terminal
printf '{"id":1,"method":"list_tools"}\n' | python mcp/scene_manager/server.py
```

The server keeps a shared SQLAlchemy engine and SentenceTransformer cache via `service.py`, so embeddings stay consistent regardless of interface. Responses follow a lightweight JSON-RPC contract (`initialize`, `list_tools`, `call_tool`, `shutdown`).

**Tool registration**: `config/tools/scene_manager_mcp.json` declares the command/env wiring so Friday's gateway (and local MCP clients) can auto-discover the scene manager. Update the values or add secrets management before production deployment.

**Smoke test**: `scripts/test_scene_manager_mcp.py` loads `.env`, spawns the server, runs `initialize`, `list_tools`, a sample `scene_search`, and then shuts it down—useful after schema or embedding updates. If LAPACK/BLAS is missing, the server now degrades to a `SequenceMatcher` fallback so the test completes (score quality drops and a warning is logged); once SciPy is installed the semantic model takes over automatically.

**Gateway helpers**:

- `scripts/mcp_gateway_probe.py` – sanity check that the gateway config lists all tools.
- `scripts/record_mcp_traces.py` – records live JSON-RPC transcripts; current output lives in `data/traces/iteration2_live_traces.jsonl` (reorder, backlog/link, revision-confirm flows).
- `data/instructions/iteration2_train.jsonl` combines Iteration 1 SFT data with the new MCP traces (12 additions) ready for fine-tuning.
- `src/training/vscode_sagemaker_trainer.py` now uploads the Iteration 2 dataset, copies the latest training entrypoint, and is ready to launch the SageMaker run.
- `scripts/run_iter2_smoke.py` passes 20/20 cases with the heuristic Friday baseline (serves as regression check pre/post fine-tune).

---

## 📊 **Current Data Status**

### **Aa Janta Naduma Project**

- **Project**: `slug="aa-janta-naduma"`, `status="draft"`
- **Scenes**: 32 total (SCN001 - SCN032)
  - **Active**: 26 scenes with content
  - **Backlog**: 6 placeholder scenes (SCN024-SCN029)
- **Embeddings**: All scenes vectorized with multilingual model
- **Sequence**: Linear chain SCN001→SCN002→...→SCN032

### **Sample Scene Structure**

```sql
-- Example: Proposal scene
scene_code: "SCN015"
title: "Proposal Scene"
canonical_text: "FADE IN: EXT. COFFEE SHOP - EVENING\n[Telugu dialogue content]..."
narrative_order: 15.0
status: "active"
tags: ["romance", "outdoor", "evening"]
```

---

## 🎯 **Iteration 2 Goals & Implementation Plan**

### **Phase 1: Complete MCP Integration** 🔄

1. **Replace FastAPI** with proper MCP server implementation
2. **Tool Registration** in Friday's agent configuration
3. **Schema Validation** for tool inputs/outputs
4. **Authentication** (localhost token for development)

### **Phase 2: Fine-tuning Data Preparation** 📝

1. **Training Examples**: Script realistic scene editing conversations

   ```json
   {
     "messages": [
       { "role": "user", "content": "Let's modify the proposal scene" },
       {
         "role": "assistant",
         "content": "I'll search for the proposal scene first.",
         "tool_calls": [{ "type": "scene_search", "query": "proposal scene" }]
       },
       { "role": "tool", "content": "[Scene SCN015 found...]" },
       {
         "role": "assistant",
         "content": "Found the proposal scene (SCN015). What changes would you like to make?"
       }
     ]
   }
   ```

2. **Multilingual Examples**: English queries → Telugu scene content
3. **Scene Operations**: Reordering, status changes, revision tracking
4. **Confirmation Patterns**: "Moving SCN031 after SCN005, done ✓"

### **Phase 3: AWS Deployment & Testing** 🚀

1. **QLoRA Fine-tune**: Same ml.g5.12xlarge process as Iteration 1
2. **Tool Integration**: MCP server accessible from SageMaker endpoint
3. **Live Testing**: Scene editing conversations with database mutations
4. **Cost Management**: Deploy → Test → Delete (no persistent endpoint)

---

## 🔍 **Technical Implementation Notes**

### **Database Connection**

```python
# Connection via db/config.py
DATABASE_URL = "postgresql://user:password@localhost:5432/friday"

# Schema creation
from db.utils import create_all
create_all()  # Materializes all tables
```

### **Embedding Update Process**

```bash
# Regenerate embeddings after scene edits
PYTHONPATH=src python scripts/update_scene_embeddings.py --recompute
```

### **Scene Import Script**

```python
# Import new scripts into database
from src.data_processing.screenplay_import import import_script
import_script("data/film/scripts/new_script.md", project_slug="new-project")
```

---

## 📋 **Dependencies Added**

### **New Requirements**

```txt
# Vector search
sentence-transformers==2.7.0

# MCP server foundation
fastapi>=0.100.0
uvicorn>=0.20.0
mcp-tools>=0.1.0

# Database enhancements
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0

# Text processing
wordfreq>=3.0.0  # For Telugu transliteration
```

### **Installation Status**

```bash
# Confirmed installed in friday_ft environment
conda install -c pytorch -c huggingface -c conda-forge sentence-transformers=2.7.0 ✅
pip install fastapi uvicorn ✅
pip install wordfreq ✅
pip install mcp-tools ✅
python -m pip install "psycopg[binary]" ✅
```

---

## 🚨 **Known Limitations & Next Steps**

### **Current Gaps**

1. **Agent Training**: No fine-tune data with tool usage examples
2. **Production Auth**: Only localhost development setup
3. **Performance**: No indexing on embedding vectors yet
4. **MCP Client Wiring**: Need to register the new server with the gateway/tooling layer

### **Immediate Actions Needed**

1. **Generate training data** with realistic tool usage patterns
2. **Test vector search** performance with larger scene databases
3. **Validate multilingual** embedding quality (Telugu/English)
4. **Hook MCP server** into Friday's agent tool registry

### **Future Enhancements**

1. **TTS Integration**: Link scene revisions to voice synthesis jobs
2. **Storyboard Hooks**: Scene ID → visual generation triggers
3. **Collaborative Editing**: Multi-user scene revision tracking
4. **Export Formats**: Final Draft, Fountain, PDF screenplay export

---

## 📖 **Reference Links**

- **Previous KT**: `FRIDAY_AI_KT_DOCUMENT.md` (Iteration 1 foundation)
- **Database Schema**: `db/schema.py` (complete table definitions)
- **MCP Server**: `mcp/scene_manager/service.py` (FastAPI implementation)
- **Embedding Script**: `scripts/update_scene_embeddings.py` (vector generation)
- **Sample Data**: `data/film/scripts/aa_janta_naduma_draft.md` (32 scenes imported)

---

## 🧾 **Iteration 2 Work Log**

### 2025-09-28

- Reviewed `previous_chat.txt` to recover Iteration 1 closing context: schema upgrades (narrative ordering, SceneLink), multilingual embeddings, and pending MCP conversion.
- Confirmed target tables to inspect (`script_scene_embeddings`, `scene_links`, `script_scenes`) before wiring a proper MCP server.
- Re-aligned immediate goals: finish MCP tooling, prep fine-tune prompts with tool calls, plan transient AWS deployment for validation.
- Outstanding checks: run a live DB inspection (using credentials from `.env`) and validate embedding integrity prior to agent integration.
- Established direct Postgres connectivity (psycopg3) and verified current data health: 1 project, 32 scenes, 32 embeddings (multilingual-mpnet), 31 sequential links; spot-checked early-scene metadata lengths.
- Chose integration strategy: keep `mcp/scene_manager/service.py` as shared core logic and add a dedicated `mcp/scene_manager/server.py` MCP host exposing `scene_search`, `scene_get`, `scene_update`, `scene_reorder`, and `scene_link` tools with cached embedding model + pooled SQLAlchemy engine.
- Implemented the MCP server (`mcp/scene_manager/server.py`) with JSON-RPC plumbing, before/after reorder heuristics, link creation, and shared embedding/DB layers; documented run instructions and tool schemas.
- Added `mcp-tools` to the Python requirements and documented installation so the MCP server entry points are available in all environments.
- Added MCP gateway mapping (`config/tools/scene_manager_mcp.json`) so the agent can spawn the server with repo-relative paths and inherited DB credentials.
- Seeded `data/instructions/iteration2_tool_examples.jsonl` with five exemplar conversations covering success/error paths across `scene_search`, `scene_get`, `scene_reorder`, `scene_update`, and `scene_link` operations.
- Added `scripts/test_scene_manager_mcp.py` to validate MCP server startup and basic tool calls after environment changes.
- Hardened `mcp/scene_manager/server.py` and `service.py` to set `sys.path` automatically and fail gracefully when `sentence-transformers` dependencies are missing.
- Expanded the tool example corpus with multi-step Telugu flows (`scene_multi_reorder`, `scene_status_check`) and an error-handled reorder case.
- Ran the MCP smoke test: initialize/list succeed; `scene_search` currently returns a dependency warning (`sentence-transformers` requires SciPy/LAPACK). Logged guidance to finish the math libs before semantic queries.
- Added a pure-Python fallback (`difflib.SequenceMatcher`) so `scene_search` keeps working even when the embedding stack is unavailable; the smoke test now returns low-confidence results instead of hard errors.
- Installed SciPy/Scikit-learn in the Anaconda runtime, restored the full embedding path with `paraphrase-multilingual-mpnet-base-v2`, and verified the smoke test returns real cosine scores.
- Captured live MCP traces via `scripts/record_mcp_traces.py` (reorder, backlog/link, revision confirm, revision revert, status toggle) for Iteration 2 fine-tune seeding.
- Merged all tool transcripts into `data/instructions/iteration2_train.jsonl` (254 rows) and updated the SageMaker trainer to upload the new dataset and copy the latest entrypoint.
- Ran `scripts/run_iter2_smoke.py` (20/20 pass) to baseline the heuristic Friday prior to fine-tune.
- Built `scripts/mcp_gateway_probe.py` for fast tool discovery checks.
- `data/traces/iteration2_live_traces.jsonl` now stores five end-to-end transcripts: reorder, backlog/link, revision confirm, revision revert, and status toggle (active ↔ backlog) — all captured against the live MCP server, returning the database to its original state.

---

_This document represents the state as of 28 September 2025. The foundation is solid - database schema complete, vector search working, FastAPI service operational. The final step is true MCP integration for seamless agent tool usage._

**Next milestone**: Complete MCP conversion → Generate training data → Fine-tune Iteration 2 → Deploy & Test → TTS Integration
