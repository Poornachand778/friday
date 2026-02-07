# Friday AI - Visual Architecture

> Complete system architecture with implementation status.
> Uses Mermaid diagrams (VS Code: `Markdown Preview Mermaid Support` extension).

**Last Updated**: February 7, 2026
**Branch**: iteration_5

---

## 1. High-Level System Overview

```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        Voice["Mic + Wake Word\n(OpenWakeWord)"]
        Text["Text Chat\n(FastAPI)"]
        Camera["Camera Trigger\n(Future)"]
    end

    subgraph Core["Friday Core"]
        Orchestrator["Orchestrator\n(orchestrator/core.py)"]
        Router["GLM-4.7-Flash Router\n(inference/router.py)"]
        LLM["LLaMA 3.1 8B\n+ Telugu LoRA\n+ Persona LoRA"]
    end

    subgraph Memory["Memory System"]
        WM["Working Memory\nContext Window Manager\n(memory/layers/working.py)"]
        STM["Short-Term Memory\n7-day SQLite\n(memory/layers/short_term.py)"]
        LTM["Long-Term Memory\nVector Embeddings\n(memory/layers/long_term.py)"]
        KG["Knowledge Graph\nTriplet Store\n(memory/layers/knowledge_graph.py)"]
        Profile["Profile Store\nBoss Identity\n(memory/layers/profile.py)"]
    end

    subgraph Tools["MCP Tool Servers (stdio)"]
        Scene["Scene Manager\n(mcp/scene_manager/)"]
        DocMCP["Document Processor\n(mcp/documents/)"]
        Email["Gmail MCP\n(mcp/gmail/)"]
        VoiceMCP["Voice MCP\n(mcp/voice/)"]
    end

    subgraph Documents["Document Understanding"]
        OCR["DeepSeek-OCR 2\n(documents/ocr/)"]
        Chunker["Semantic Chunker\n(documents/pipeline/)"]
        BookComp["Book Comprehension\n(documents/understanding/)"]
        Search["Hybrid Search\nBM25 + Semantic\n(documents/retrieval/)"]
    end

    subgraph VoicePipe["Voice Pipeline"]
        STT["Faster-Whisper\n(voice/stt/)"]
        TTS["Chatterbox-Turbo EN\nIndicF5 Telugu\n(voice/tts/)"]
        VAD["Voice Activity Detection\n(voice/audio/vad.py)"]
        WakeWord["OpenWakeWord\n'Hey Friday'\n(voice/wakeword/)"]
    end

    subgraph VectorDB["Vector Databases"]
        PGVector["pgvector\n(SQL + Vector)"]
        Qdrant["Qdrant\n(RAG + Filtering)"]
    end

    subgraph Output["Output Layer"]
        VoiceOut["Voice Response\n(TTS Stream)"]
        TextOut["Text Response\n(HTTP/WebSocket)"]
    end

    Voice --> WakeWord --> VAD --> STT --> Orchestrator
    Text --> Orchestrator
    Camera -.-> Orchestrator

    Orchestrator --> Router
    Router --> LLM

    Orchestrator <--> WM
    WM <--> STM
    STM --> LTM
    LTM <--> KG
    Profile --> WM

    Orchestrator <-->|"MCP stdio"| Scene
    Orchestrator <-->|"MCP stdio"| DocMCP
    Orchestrator <-->|"MCP stdio"| Email
    Orchestrator <-->|"MCP stdio"| VoiceMCP

    DocMCP --> OCR --> Chunker --> BookComp
    Chunker --> Search

    LTM <--> Qdrant
    Scene <--> PGVector
    Search <--> Qdrant

    LLM --> TTS --> VoiceOut
    LLM --> TextOut
```

---

## 2. Agentic Routing Flow (Step-by-Step)

```mermaid
sequenceDiagram
    participant U as User (Voice/Text)
    participant O as Orchestrator
    participant R as GLM Router
    participant M as Memory System
    participant L as LLaMA 8B + LoRA
    participant T as MCP Tools
    participant Out as Output (TTS/Text)

    U->>O: "Friday, courtroom scene lo tension missing"

    Note over O: Step 1: Context Detection
    O->>O: Detect context (Writers Room)
    O->>O: Detect language (Telugu-English mixed)

    Note over O,M: Step 2: Memory Prefetch (parallel)
    O->>M: Query Working Memory (recent turns)
    O->>M: Query LTM ("courtroom" + "tension")
    O->>M: Query KG (courtroom scene entities)
    M-->>O: Prefetched context bundle

    Note over O,R: Step 3: Route Decision
    O->>R: Should I use tools? Which ones?
    R-->>O: YES: scene_search("courtroom")

    Note over O,T: Step 4: Tool Execution
    O->>T: scene_search("courtroom", project="current")
    T-->>O: Found: ACT2_SC15 - Cross-examination scene

    Note over O,L: Step 5: LLM Generation
    O->>L: System prompt + memory context + tool results + user message
    L-->>O: Response with diagnosis (streaming)

    Note over O,Out: Step 6: Output
    O->>Out: Stream response (text or TTS)

    Note over O,M: Step 7: Memory Storage (background)
    O->>M: Store turn in Working Memory
    O->>M: Extract facts to STM
    O->>M: Update attention stack
```

---

## 3. Memory Integration Architecture

```mermaid
flowchart TB
    subgraph Orchestrator["Orchestrator (orchestrator/core.py)"]
        AgentLoop["Agent Loop\nmax 5 iterations"]
        ContextBuilder["Context Builder\nbuild_prompt()"]
    end

    subgraph WorkingMem["Working Memory (ACTIVE)"]
        TokenCounter["Token Counter\ntiktoken / fallback"]
        PoisonDetect["Context Poisoning\nDetector"]
        HybridBuffer["Hybrid Buffer\n20% compressed\n60% verbatim\n20% reserve"]
        AttentionStack["Attention Stack\n7 items max"]
    end

    subgraph Capacity["Capacity Zones"]
        Normal["Normal < 70%\nNo action"]
        Proactive["Proactive 70-85%\nSummarize 1 turn"]
        Aggressive["Aggressive 85-95%\nSummarize 3 turns"]
        Emergency["Emergency > 95%\nEmergency prune"]
    end

    subgraph STMLayer["Short-Term Memory (7 days)"]
        SQLiteFTS["SQLite + FTS5"]
        FactExtract["Fact Extraction\n(GLM-4.7-Flash)"]
        DualTimestamp["Dual Timestamps\ndocument_time\nevent_time"]
    end

    subgraph LTMLayer["Long-Term Memory (Permanent)"]
        VectorStore["Vector Embeddings\n768-dim"]
        DecayEngine["Decay Algorithm\nscore = 0.3*recency\n+ 0.15*frequency\n+ 0.3*importance\n+ 0.15*event\n+ 0.1*profile"]
        MemTypes["Types: FACT\nPREFERENCE | EVENT\nPATTERN | DECISION\nRELATIONSHIP"]
    end

    subgraph KGLayer["Knowledge Graph"]
        Triplets["Subject-Relation-Object\n(NetworkX backend)"]
        HybridSearch["Hybrid Search\nVector + Graph"]
    end

    subgraph ProfileLayer["Profile (Never Decays)"]
        StaticID["Static: Name, Role\nLanguages, Style"]
        DynamicState["Dynamic: Current project\nRoom, Mood"]
        Preferences["Learned Preferences\n(confidence > 0.8)"]
    end

    AgentLoop --> ContextBuilder
    ContextBuilder --> HybridBuffer
    ContextBuilder --> AttentionStack

    HybridBuffer --> TokenCounter
    TokenCounter --> Normal & Proactive & Aggressive & Emergency
    HybridBuffer --> PoisonDetect

    HybridBuffer -->|"Session end"| FactExtract
    FactExtract --> SQLiteFTS
    SQLiteFTS --> DualTimestamp

    SQLiteFTS -->|"Consolidate\n(nightly)"| VectorStore
    VectorStore --> DecayEngine
    VectorStore --> MemTypes

    VectorStore -->|"Extract entities"| Triplets
    Triplets --> HybridSearch
    HybridSearch -->|"Prefetch"| AttentionStack

    ProfileLayer --> ContextBuilder

    style Normal fill:#22c55e,color:#000
    style Proactive fill:#eab308,color:#000
    style Aggressive fill:#f97316,color:#000
    style Emergency fill:#ef4444,color:#fff
```

---

## 4. Document Processing Pipeline (DeepSeek OCR)

```mermaid
flowchart LR
    subgraph Input["Document Input"]
        PDF["PDF / Image\n(Book, Script, Reference)"]
    end

    subgraph OCR["DeepSeek-OCR 2 (3B)"]
        PageProcess["Process Page\nVisual Causal Flow"]
        LayoutDetect["Layout Detection\nMulti-column, Tables"]
        TeluguOCR["Telugu Script\nRecognition"]
    end

    subgraph Pipeline["Processing Pipeline"]
        Chunker["Semantic Chunker\n(documents/pipeline/chunker.py)"]
        Embedder["Embedding Generator\nparaphrase-multilingual-mpnet"]
        TripletEx["Triplet Extractor\n(memory/operations/)"]
        Citation["Citation Tracker\nBook > Chapter > Page"]
    end

    subgraph Storage["Storage Layer"]
        DocStore["Document Store\n(SQLite)"]
        VecStore["Vector Store\n(Qdrant)"]
        KGStore["Knowledge Graph\n(NetworkX)"]
        CompStore["Comprehension Store\n(understanding_store.py)"]
    end

    subgraph Query["Query Interface"]
        AskBook["ask_about_book()\n'What does Syd Field\nsay about Act 2?'"]
        FindRef["find_reference()\n'Courtroom scene\ntechniques'"]
        BrainStorm["brainstorm_with()\nUse book knowledge\nin conversation"]
    end

    PDF --> PageProcess
    PageProcess --> LayoutDetect
    PageProcess --> TeluguOCR
    LayoutDetect --> Chunker
    TeluguOCR --> Chunker

    Chunker --> Embedder --> VecStore
    Chunker --> TripletEx --> KGStore
    Chunker --> Citation --> DocStore
    Chunker --> CompStore

    VecStore --> AskBook
    KGStore --> FindRef
    DocStore --> BrainStorm
    CompStore --> BrainStorm
```

**This is CRITICAL for you**: Process screenwriting books, reference scripts, and research material so Friday can brainstorm with that knowledge.

---

## 5. Voice Pipeline Flow

```mermaid
flowchart LR
    subgraph Capture["Audio Capture"]
        Mic["Microphone"]
        WW["OpenWakeWord\n'Hey Friday'"]
        VAD2["VAD\nSpeech Detection"]
    end

    subgraph STTBlock["Speech-to-Text"]
        Whisper["Faster-Whisper\nlarge-v3"]
        LangDetect["Language Detector\nEN / TE / Mixed"]
    end

    subgraph Process["Friday Core"]
        Orch2["Orchestrator"]
        LLM2["LLaMA 8B\n+ LoRAs"]
    end

    subgraph TTSBlock["Text-to-Speech (Hybrid)"]
        LangRouter["Language Router"]
        Chatterbox["Chatterbox-Turbo\nEnglish/Hindi\nsub-200ms"]
        IndicF5["IndicF5\nTelugu\nNative support"]
        VoiceClone["Boss Voice Clone\n5-10s reference"]
    end

    subgraph Play["Audio Output"]
        Speaker["Speaker"]
    end

    Mic --> WW -->|"Wake detected"| VAD2
    VAD2 -->|"Speech chunk"| Whisper
    Whisper --> LangDetect
    LangDetect --> Orch2
    Orch2 --> LLM2
    LLM2 -->|"Response text"| LangRouter

    LangRouter -->|"EN/HI tokens"| Chatterbox
    LangRouter -->|"TE tokens"| IndicF5

    VoiceClone --> Chatterbox
    VoiceClone --> IndicF5

    Chatterbox --> Speaker
    IndicF5 --> Speaker
```

### Latency Target

```mermaid
gantt
    title Voice Response Timeline (Target < 800ms to first audio)
    dateFormat X
    axisFormat %Lms

    section Pipeline
    Wake Word Detection    :done, 0, 50
    VAD + Capture          :done, 50, 150
    STT (Whisper)          :active, 150, 350
    LLM Prefill            :active, 350, 500
    First Token Decode     : 500, 600
    TTS First Chunk        : 600, 800
    Audio Playback Starts  :milestone, 800, 800
```

---

## 6. Training Pipeline (Updated)

```mermaid
flowchart TB
    subgraph Phase1["Phase 1: Interview Data (DONE)"]
        Interviews["120 Interview\nExchanges"]
        WhatsApp["350 Curated\nWhatsApp"]
        Tools["30 Tool\nExamples"]
        Contra["25 Contrastive\nPairs"]
    end

    subgraph Phase2["Phase 2: Behavioral Data (IN PROGRESS)"]
        Inv["INVESTIGATOR\n3/40 done"]
        Critic["CRITIC\n0/35 done"]
        Story["STORYTELLER\n0/40 done"]
        Brain["BRAINSTORM\n0/35 done"]
    end

    subgraph Transform["Transform"]
        Quality["TDQO Review\nQuality Score 1-5"]
        Format["ChatML Format\nrole: user/assistant"]
        Validate["Schema Validation\nTool format check"]
    end

    subgraph TrainBlock["Training (DGX Spark)"]
        TeluguLoRA["Telugu LoRA\nIndicAlign data\nr=64, alpha=128"]
        PersonaLoRA["Persona LoRA\nPhase 2 data\nr=64, alpha=128"]
        Merge["LoRA Merge\nbase + telugu + persona"]
    end

    subgraph Deploy["Deployment"]
        vLLM["vLLM Server\n4-bit quantization"]
        Friday["Friday Model\nTelugu-fluent\nBoss's thinking patterns"]
    end

    Phase1 --> Quality
    Phase2 --> Quality
    Quality --> Format --> Validate

    Validate --> TeluguLoRA
    Validate --> PersonaLoRA
    TeluguLoRA --> Merge
    PersonaLoRA --> Merge
    Merge --> vLLM --> Friday

    style Inv fill:#eab308,color:#000
    style Critic fill:#ef4444,color:#fff
    style Story fill:#ef4444,color:#fff
    style Brain fill:#ef4444,color:#fff
```

---

## 7. DGX Spark Deployment Layout

```mermaid
block-beta
    columns 4

    block:header:4
        columns 1
        title["DGX Spark - 128GB Unified Memory"]
    end

    block:model1:1
        columns 1
        m1title["Friday LLM"]
        m1detail["LLaMA 3.1 8B\n+ Telugu LoRA\n+ Persona LoRA\n~16GB FP16"]
    end

    block:model2:1
        columns 1
        m2title["Storyboard"]
        m2detail["FLUX 12B\nComfyUI\n~24GB FP8"]
    end

    block:model3:1
        columns 1
        m3title["Kitchen"]
        m3detail["Qwen2.5-VL 7B\n+ YOLOv8\n~10GB"]
    end

    block:model4:1
        columns 1
        m4title["Voice Pipeline"]
        m4detail["Whisper STT\n+ Chatterbox\n+ IndicF5\n~8GB"]
    end

    block:remaining:4
        columns 1
        free["REMAINING: ~70GB for context, KV cache, future models"]
    end

    block:services:4
        columns 4
        svc1["vLLM Server"]
        svc2["ComfyUI"]
        svc3["Qdrant Docker"]
        svc4["PostgreSQL\n+ pgvector"]
    end

    style model1 fill:#3b82f6,color:#fff
    style model2 fill:#8b5cf6,color:#fff
    style model3 fill:#22c55e,color:#000
    style model4 fill:#f97316,color:#000
    style remaining fill:#1e293b,color:#94a3b8
```

---

## 8. MCP Tool Routing Architecture

```mermaid
flowchart TB
    subgraph Orch["Orchestrator"]
        ToolRegistry["Tool Registry\n(orchestrator/tools/registry.py)"]
        ToolExec["Tool Executor"]
    end

    subgraph MCPServers["MCP Servers (stdio)"]
        subgraph SceneMCP["Scene Manager MCP"]
            SceneSearch["scene_search()"]
            SceneGet["scene_get()"]
            SceneUpdate["scene_update()"]
            SceneReorder["scene_reorder()"]
            SceneLink["scene_link()"]
        end

        subgraph DocsMCP2["Documents MCP"]
            DocIngest["doc_ingest()"]
            DocSearch2["doc_search()"]
            DocAsk["doc_ask()"]
        end

        subgraph GmailMCP["Gmail MCP"]
            EmailRead["email_read()"]
            EmailSearch["email_search()"]
            EmailDraft["email_draft()"]
        end

        subgraph VoiceMCP2["Voice MCP"]
            VoiceRecord["voice_record()"]
            VoicePlay["voice_play()"]
            VoiceClone2["voice_clone()"]
        end
    end

    subgraph Storage2["Storage Backends"]
        SceneDB["Scene DB\nPostgreSQL + pgvector"]
        DocDB["Document Store\nSQLite"]
        MemDB["Memory Store\nQdrant + SQLite"]
    end

    ToolRegistry -->|"JSON-RPC\nstdio"| SceneMCP
    ToolRegistry -->|"JSON-RPC\nstdio"| DocsMCP2
    ToolRegistry -->|"JSON-RPC\nstdio"| GmailMCP
    ToolRegistry -->|"JSON-RPC\nstdio"| VoiceMCP2

    SceneMCP --> SceneDB
    DocsMCP2 --> DocDB
    DocsMCP2 --> MemDB
```

---

## 9. Component Implementation Status

```mermaid
pie title Overall Friday Development Status (Feb 2026)
    "Built and Working" : 40
    "Built, Needs Integration" : 25
    "Partially Built" : 15
    "Not Started" : 20
```

### Detailed Component Status

| Component | Status | % | Key File | Blocker |
|-----------|--------|---|----------|---------|
| **Orchestrator Core** | Built | 75% | `orchestrator/core.py` | Memory integration |
| **GLM Router** | Built | 90% | `orchestrator/inference/router.py` | Upskill enhancement |
| **Working Memory** | Built | 95% | `memory/layers/working.py` | LLM summarizer |
| **STM (7-day)** | Built | 90% | `memory/layers/short_term.py` | - |
| **LTM (Vector)** | Built | 90% | `memory/layers/long_term.py` | pgvector/Qdrant |
| **Knowledge Graph** | Built | 85% | `memory/layers/knowledge_graph.py` | Triplet extractor |
| **Profile Store** | Built | 80% | `memory/layers/profile.py` | - |
| **Scene Manager MCP** | Built | 95% | `mcp/scene_manager/` | - |
| **Gmail MCP** | Built | 60% | `mcp/gmail/` | OAuth testing |
| **Documents Module** | Partial | 70% | `documents/manager.py` | OCR testing |
| **DeepSeek OCR** | Structure | 30% | `documents/ocr/` | Not tested |
| **Book Comprehension** | Built | 70% | `documents/understanding/` | OCR + LTM link |
| **Voice Daemon** | Partial | 40% | `voice/daemon.py` | STT/TTS config |
| **Faster-Whisper STT** | Structure | 20% | `voice/stt/` | Not tested |
| **TTS (Chatterbox)** | Not started | 0% | `voice/tts/` | Replace XTTS |
| **Wake Word** | Structure | 20% | `voice/wakeword/` | Training samples |
| **Training Data (Phase 1)** | Done | 100% | `data/instructions/` | - |
| **Training Data (Phase 2)** | Started | 2% | `data/phase2/` | Boss playing Friday |
| **Model Training** | Not started | 0% | - | Phase 2 data + DGX |
| **Vector DB Setup** | Not started | 0% | - | Install pgvector/Qdrant |
| **Telugu LoRA** | Not started | 0% | - | IndicAlign download |
| **FastAPI Server** | Built | 90% | `orchestrator/main.py` | - |
| **Tests** | Minimal | 10% | `tests/` | Coverage expansion |

---

## 10. Critical Integration Map

```mermaid
flowchart LR
    subgraph Done["BUILT (Not Connected)"]
        A["Working Memory\n(memory/layers/working.py)"]
        B["Orchestrator\n(orchestrator/core.py)"]
        C["Scene Manager MCP\n(mcp/scene_manager/)"]
        D["LTM Vector Search\n(memory/layers/long_term.py)"]
        E["Document Chunker\n(documents/pipeline/)"]
    end

    subgraph Gaps["INTEGRATION GAPS"]
        G1["Gap 1:\nOrchestrator uses\nsimple ConversationMemory\nnot Working Memory"]
        G2["Gap 2:\nTools call locally\nnot through MCP stdio"]
        G3["Gap 3:\nDocument chunks\nnot stored in LTM"]
        G4["Gap 4:\nVoice daemon\nnot connected to\nOrchestrator"]
        G5["Gap 5:\nKG not queried\nduring retrieval"]
    end

    A -.->|"NOT CONNECTED"| G1
    G1 -.-> B

    C -.->|"NOT ROUTED"| G2
    G2 -.-> B

    E -.->|"NOT STORED"| G3
    G3 -.-> D

    style G1 fill:#ef4444,color:#fff
    style G2 fill:#ef4444,color:#fff
    style G3 fill:#ef4444,color:#fff
    style G4 fill:#ef4444,color:#fff
    style G5 fill:#ef4444,color:#fff
```

---

## 11. Future Extensibility (Storyboard + Kitchen)

```mermaid
flowchart TB
    subgraph Friday["Friday Core (Script Writing)"]
        FridayLLM["LLaMA 8B + LoRAs"]
        FridayMem["Memory System"]
        FridayTools["MCP Tools"]
    end

    subgraph Storyboard["Storyboard Module (Future)"]
        FLUX["FLUX 12B\nImage Generation"]
        ComfyUI["ComfyUI\nWorkflow Engine"]
        SceneToImage["Scene Description\n--> Visual Prompt"]
        LoRAStyles["Style LoRAs\n(Director's vision)"]
    end

    subgraph Kitchen["Kitchen Module (Future)"]
        KitchVision["Qwen2.5-VL 7B\nVision-Language"]
        YOLO["YOLOv8\nFood Detection"]
        RecipeDB["Recipe Database\n(RAG)"]
        KitchVoice["Voice Interaction\n(shared pipeline)"]
    end

    subgraph Shared["Shared Infrastructure"]
        VoicePipe["Voice Pipeline\n(Whisper + TTS)"]
        VectorDBs["Vector Databases\n(pgvector + Qdrant)"]
        DGXSpark["DGX Spark\n128GB Unified Memory"]
    end

    FridayLLM -->|"Scene description"| SceneToImage
    SceneToImage --> FLUX
    FLUX --> ComfyUI
    LoRAStyles --> FLUX

    FridayLLM -->|"Room: Kitchen"| KitchVision
    KitchVision --> YOLO
    RecipeDB --> KitchVision

    Friday --> Shared
    Storyboard --> DGXSpark
    Kitchen --> DGXSpark
    VoicePipe --> KitchVoice

    style Friday fill:#3b82f6,color:#fff
    style Storyboard fill:#8b5cf6,color:#fff
    style Kitchen fill:#22c55e,color:#000
```

---

## 12. What to Build Next (Priority Order)

```mermaid
flowchart TB
    subgraph Now["CAN DO NOW (No Boss needed)"]
        T1["1. Integrate Working Memory\ninto Orchestrator"]
        T2["2. Wire MCP tool routing"]
        T3["3. Test DeepSeek OCR\nwith sample PDFs"]
        T4["4. Setup pgvector\n+ Qdrant Docker"]
        T5["5. Generate tool\ntraining examples"]
    end

    subgraph Boss["NEEDS BOSS"]
        B1["6. Phase 2 conversations\n(147 remaining)"]
        B2["7. Voice samples\n(Telugu + English)"]
        B3["8. Review transformed\ninterview data"]
    end

    subgraph DGX["WHEN DGX ARRIVES"]
        D1["9. Train Telugu LoRA"]
        D2["10. Train Persona LoRA"]
        D3["11. Merge + Deploy"]
        D4["12. Voice pipeline\nend-to-end test"]
    end

    T1 --> T2
    T3 --> T4
    B1 --> D2
    D1 --> D3
    D2 --> D3
    D3 --> D4

    style T1 fill:#22c55e,color:#000
    style T2 fill:#22c55e,color:#000
    style T3 fill:#22c55e,color:#000
    style T4 fill:#22c55e,color:#000
    style T5 fill:#22c55e,color:#000
    style B1 fill:#eab308,color:#000
    style B2 fill:#eab308,color:#000
    style B3 fill:#eab308,color:#000
    style D1 fill:#3b82f6,color:#fff
    style D2 fill:#3b82f6,color:#fff
    style D3 fill:#3b82f6,color:#fff
    style D4 fill:#3b82f6,color:#fff
```

---

## Quick Reference: File Locations

| Component | Key Files |
|-----------|-----------|
| **Orchestrator** | `orchestrator/core.py`, `orchestrator/main.py` |
| **GLM Router** | `orchestrator/inference/router.py` |
| **LLM Client** | `orchestrator/inference/local_llm.py` |
| **Working Memory** | `memory/layers/working.py` |
| **Memory Manager** | `memory/manager.py` |
| **STM** | `memory/layers/short_term.py` |
| **LTM** | `memory/layers/long_term.py` |
| **Knowledge Graph** | `memory/layers/knowledge_graph.py` |
| **Profile** | `memory/layers/profile.py` |
| **Telugu Processor** | `memory/telugu/processor.py` |
| **Scene Manager** | `mcp/scene_manager/service.py` |
| **Documents** | `documents/manager.py` |
| **DeepSeek OCR** | `documents/ocr/deepseek_engine.py` |
| **Book Comprehension** | `documents/understanding/comprehension.py` |
| **Voice Daemon** | `voice/daemon.py` |
| **STT** | `voice/stt/faster_whisper_service.py` |
| **TTS** | `voice/tts/xtts_service.py` (to be replaced) |
| **Config** | `config/orchestrator_config.yaml` |
| **Training Data** | `data/phase2/behavioral_conversations/` |
| **Phase 2 Guide** | `prompts/phase2_data_generation_guide.md` |

---

## Research Documents Index

| Research | File | Key Finding |
|----------|------|-------------|
| **Telugu LoRA** | `docs/research/telugu_lora_adapter_stacking_research.md` | Doable, use IndicAlign + merge |
| **Vector Databases** | `docs/research/vector_databases_research.md` | pgvector + Qdrant dual approach |
| **Chatterbox TTS** | `docs/research/chatterbox_tts_research.md` | Replace XTTS v2, hybrid EN+TE |
| **DGX Spark** | `docs/research/dgx_spark_compatibility_research.md` | Suitable, all 4 models fit |
| **Memory Architecture** | `docs/architecture/FRIDAY_MEMORY_ARCHITECTURE.md` | Brain-inspired, 8 layers |
| **Context Window** | `docs/CONTEXT_WINDOW_MANAGEMENT.md` | 70% threshold, hybrid buffer |

---

## View This Document

1. Install `Markdown Preview Mermaid Support` extension in VS Code
2. Open this file and press `Cmd+Shift+V` for preview
3. Or use `Markdown Preview Enhanced` for side-by-side view

---

*"The architecture is the skeleton. Training data is the soul. Integration is the nervous system."*
