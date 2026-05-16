# Tech Decomposition & Bulletproof Grounding

An enterprise-grade, multi-repository AI coding assistant framework. It turns natural language queries or Jira tickets into **highly-accurate technical decompositions** and answers, using a custom "Bulletproof Grounding" pipeline that eliminates LLM hallucination across complex codebases.

## 🚀 High-Level Architecture

The system is split into two primary components that communicate to provide the ultimate AI developer experience:

```mermaid
graph TD
    User([User / Jira]) --> PythonBackend
    
    subgraph PythonBackend [Python Orchestrator (tech-decomposition)]
        Enricher[LLM Query Enricher<br/>Gemini Flash]
        Grounding[Bulletproof Grounding Engine]
        Adapters[Adapter Registry<br/>Gemini, OpenAI, etc.]
        
        Enricher --> Grounding
        Grounding --> Adapters
    end
    
    subgraph GroundingEngine [Bulletproof Grounding Pipeline]
        AST[Tree-sitter AST Parser]
        Chroma[(Local ChromaDB<br/>Semantic Index)]
        Sourcebot[(Remote Sourcebot<br/>RAG)]
        Inflate[Local Snippet Inflation]
        
        AST --> Chroma
        Chroma --> Inflate
        Sourcebot --> Inflate
    end
    
    subgraph AgentNode [Node.js Tool Execution Server (agent-node)]
        GenAI[Google GenAI SDK]
        OS[OS File Tools<br/>grep, find, fs]
        
        GenAI <--> OS
    end

    PythonBackend <--> AgentNode
    Grounding -.- Codebase[(Local Multi-Repo Workspace)]
```

### 1. The `agent-node` (TypeScript)
A fast, isolated Node.js server that leverages the native Automatic Function Calling (AFC) capabilities of the `@google/genai` SDK. It exposes OS-level workspace tools securely:
- `read_file`: Reads specific file contents.
- `list_directory`: Maps out folders.
- `search_files`: Uses OS `find` to locate files by pattern.
- `grep_search`: Uses OS `grep -rnI` for fast regex codebase searches.

### 2. The Python Orchestrator (`tech-decomposition`)
The primary backend that manages user intents, Jira parsing, adapter routing, and the grounding pipeline.

---

## 🛡️ The "Bulletproof Grounding" Pipeline

Standard LLMs hallucinate when asked about large, multi-repo architectures because they don't have the context. We built an **SDK-Agnostic Pre-Context Engine** that forces the LLM to read the exact, relevant code across all your repositories *before* it generates an answer.

Here is the exact flow of our grounding pipeline (`_grounding.py`):

1. **LLM Query Enrichment:** We use a cheap, fast model (Gemini Flash) via `pydantic-ai` to rewrite the user's messy question into a dense, structured query of exact code keywords and expected repos.
2. **AST-Powered Repo Mapping:** We use **Tree-sitter** to parse the local workspace (Python, JS, TS, TSX). It generates an architectural skeleton mapping every file to its specific `class_definition`, `function_declaration`, and React `component` signatures.
3. **Local Semantic Indexing (Offline RAG):** 
   - If no local index exists, we extract the exact byte-bounds of every AST function/class and embed them using **Hugging Face Sentence Transformers**.
   - These chunks are stored in a local, serverless **ChromaDB**. 
   - We execute a semantic vector search to find the top exact code chunks related to the user's intent.
4. **Enterprise Remote RAG:** We query your internal **Sourcebot** instance as a fallback/supplement to the local vector search.
5. **Local Snippet Inflation:** Whenever a chunk is retrieved, the engine opens the raw local file and "inflates" the chunk by reading 5 lines backwards and forwards, ensuring the AI sees the surrounding imports and context.

The engine concatenates this massive block of intelligence (often 80,000+ tokens) and prepends it to the LLM's prompt. 

---

## 🔌 Adapter Registry

The Python backend uses an Adapter pattern to test and execute different AI strategies. 

- `gemini`: Raw Google GenAI SDK (relies purely on iterative tool calling).
- `gemini-grounded`: Injects the Bulletproof Grounding pre-context into the Gemini SDK, resulting in 0 tool calls and 100% architectural accuracy.
- `openai_agents`: Experimental integration for OpenAI frameworks.

You can run bake-offs between these adapters to mathematically prove which approach yields the best results at the lowest cost.

---

## 📊 Evaluation & Bake-off Framework

The repository includes a strict evaluation framework to test adapter performance against real queries.

```bash
# Run a specific test case against both the raw and grounded adapters
make eval-run ADAPTERS=gemini,gemini-grounded IDS=q1-data-sharing-geolocation
```

The system automatically measures:
- **Success Rate**
- **Tokens In (Context Window)**
- **Tokens Out (Answer Length)**
- **Cost (USD)**
- **Duration**
- **Required Tool Calls**

*Example result: The grounded adapter ingested 97,000 tokens of pre-context, required 0 subsequent tool calls, and perfectly identified an asynchronous Airflow DAG handoff between 3 separate repositories.*

---

## 🛠️ Setup & Execution

### Prerequisites
- Python ≥ 3.11
- [Docker](https://docs.docker.com/desktop/) (for Sourcebot)
- Node.js (for `agent-node`)
- `uv` (recommended Python package manager)

### Local Dev Setup

1. **Environment:**
   ```bash
   cp .env.example .env
   # Add your GEMINI_API_KEY and configure REPOS_ROOT
   ```

2. **Install Python & Node Dependencies:**
   ```bash
   make install
   make ui-install
   ```

3. **Start the Stack (API, UI, Agent Node, Docker Backends):**
   ```bash
   make dev-all
   ```

### Execution

**Ask a question against your local codebases:**
```bash
# Uses the default adapter to answer a question across your local repos
make smoke
```

**Decompose a Jira Ticket (requires Jira API env vars):**
```bash
tech-decomposition --ticket-key SCRUM-17 --post-to-jira
```

---

## 📦 Directory Structure

```text
src/tech_decomposition/
├── adapters/
│   ├── _grounding.py         # The Bulletproof Grounding Engine
│   ├── _local_index.py       # ChromaDB + Sentence Transformers RAG
│   ├── _repo_map.py          # Tree-sitter AST parser
│   ├── registry.py           # Adapter registration mapping
│   ├── _gemini_grounded.py   # Grounded Adapter Implementation
│   └── _gemini.py            # Standard SDK Implementation
├── api.py                    # FastAPI entrypoint
└── enrichers/                # LLM-powered query generators

services/agent-node/
├── src/adapters/
│   ├── gemini.ts             # Google GenAI SDK integration
│   └── workspace_tools.ts    # OS-level file operations (find, grep)

eval/                         # The bake-off evaluation framework
```