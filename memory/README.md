Memory System
==============

Purpose: Single source of truth for Friday's persona, rules, long-term memories, reusable snippets, short-term session traces, and evaluation/adapter registries. All files live locally in JSON or JSONL and can be edited by tools or by hand.

Quick Start
- Add a long-term memory: `python scripts/memory_cli.py add-ltm --text "New fact" --tags film tone:decisive --lang te --trust 4`
- Search long-term memory: `python scripts/memory_cli.py search-ltm --query "New fact" --top-k 5`
- Add a snippet: `python scripts/memory_cli.py add-snippet --title "Email opener" --body "Boss, ..." --tags comms`
- Update persona from file: `python scripts/memory_cli.py set-persona --file memory/data/persona/profile.json`

Key Files
- Persona & Rules (loaded into prompts without embeddings):
  - `memory/data/persona/profile.json`
  - `memory/data/principles/rules.json`

- Long-Term Knowledge (embedded, retrievable):
  - `memory/data/ltm_memories.jsonl`
  - `memory/data/content_snippets.jsonl`

- Short-Term & Trace:
  - `memory/data/stm_sessions.jsonl`
  - `memory/data/interactions.jsonl`

- Training Datasets Registries:
  - `memory/data/sft_datasets.jsonl`
  - `memory/data/dpo_pairs.jsonl`

- Evaluation & Promotion:
  - `memory/data/eval_suites.jsonl`
  - `memory/data/eval_cases.jsonl`
  - `memory/data/eval_runs.jsonl`
  - `memory/data/eval_results.jsonl`
  - `memory/data/adapters.jsonl`

Schemas
- JSON Schema definitions for each entity live in `memory/schemas/`. The CLI performs basic shape checks.

Embeddings
- Persona and principles are used as-is (no embeddings).
- Long-term memories and content snippets support a 1024-dim `embedding` field. If absent, lexical search is used as a fallback.

Indexing
- This repo uses file-based search (lexical or vector). If you later integrate a DB, the schemas include fields compatible with IVFFLAT (vector) and GIN (tags).

Conventions
- Time fields are UTC ISO8601.
- `tags` is an array of short lowercase tokens, e.g., `film`, `checklist`, `tone:decisive`.
- `trust` is 1–5 where 5 is highly reliable.

