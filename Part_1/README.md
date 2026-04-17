# Acarnae RAG Pipeline

A minimal, production-directional RAG pipeline that ingests an AQA GCSE History exam paper
and answers curriculum-specific questions about it.

**No API keys required.** Runs entirely locally using Ollama + sentence-transformers.

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd acarnae-rag
pip install -r requirements.txt

# 2. Install Ollama (local LLM — free)
#    Mac/Linux: https://ollama.com/download
#    Then pull a model (one-time, ~4GB):
ollama pull mistral

# 3. Start Ollama (keep this terminal open)
ollama serve

# 4. In a new terminal — ingest the exam paper
python ingest.py --file data/exam_paper.txt

# 5. Run all 5 sample questions
python pipeline.py

# 6. View results
cat output/results.txt
```

Results are written to `output/results.txt` (human-readable) and `output/results.json` (machine-readable).

---

## Running a Single Question

```bash
# Run only question 2 (concept gap analysis)
python pipeline.py --question 2

# Ask your own question
python query.py "Which Assessment Objectives are tested most heavily in this paper?"

# Use your own exam paper PDF
python ingest.py --file your_paper.pdf --reset
python pipeline.py --file your_paper.pdf
```

## Using API Keys Instead of Ollama

If you have API keys, set them as environment variables — the pipeline auto-detects and uses them:

```bash
# Anthropic Claude (recommended for exam reasoning quality)
export ANTHROPIC_API_KEY=sk-ant-...
python pipeline.py

# OpenAI GPT-4o-mini
export OPENAI_API_KEY=sk-...
python pipeline.py
```

Priority order: Ollama (local) → Anthropic → OpenAI.

---

## Project Structure

```
acarnae-rag/
├── data/
│   └── exam_paper.txt      # Structured exam paper (AQA GCSE History 8145/1A/A)
├── output/
│   ├── results.txt         # Pre-run answers (human-readable)
│   └── results.json        # Pre-run answers (machine-readable)
├── chroma_db/              # Local vector store (auto-created on ingest)
├── ingest.py               # PDF → chunks → embeddings → ChromaDB
├── query.py                # RAG query engine (retrieve → prompt → LLM → answer)
├── pipeline.py             # Runs all 5 sample questions end-to-end
├── prompts.py              # All prompt templates (explicit, versioned, board-aware)
└── requirements.txt
```

---

## Stack Justification

### Why no LangChain / LlamaIndex?

This is a deliberate choice, consistent with the AI vision in Acarnae's architecture:

> "Wrappers make decisions about prompt construction, context assembly, chunking, and retries
> that you have not explicitly authorised. When a student gets a wrong answer, you cannot
> inspect exactly which tokens were sent and why."

Every step in this pipeline is explicit and inspectable:
- `ingest.py` owns the chunking logic — question boundaries, not token boundaries
- `prompts.py` owns every token of every prompt template
- `query.py` owns the retrieval and assembly logic
- There is no framework making invisible decisions between these steps

### Why Ollama (local LLM)?

- **Zero cost** — no API spend, no key required
- **Offline-capable** — consistent with Acarnae's offline-first Android requirement
- **Portable** — assessors can run this with one install command, no accounts
- **Mistral 7B** is strong enough for structured exam Q&A; `llama3.2` or `phi3` also work

### Why sentence-transformers for embeddings?

- `all-MiniLM-L6-v2` runs in ~20ms per batch on CPU, downloads once (~80MB)
- Cosine similarity in ChromaDB gives good retrieval for short exam-style queries
- Same model is used for both ingestion and query — guaranteed embedding space alignment
- No API rate limits, no cost per embedding call

### Why ChromaDB?

- Embedded (no server to start), persists to disk, zero config
- Correct choice at this scale — a separate vector DB (Pinecone, Weaviate) adds operational
  complexity for no retrieval quality benefit at <1000 chunks
- Production path: replace with pgvector on PostgreSQL when co-locating with relational data

### Why exam-aware chunking?

Generic chunkers (LangChain's `RecursiveCharacterTextSplitter`, etc.) split on token count.
This breaks the question–mark-scheme relationship:

```
Generic chunk: "...right on its side.\n\nMARK SCHEME — QUESTION 01:\nLevel 2 (3–4 marks)..."
   → The question text and its mark scheme end up in different chunks
   → Retrieval for "what does zero mean on Q01" misses the mark scheme

This pipeline: Each QUESTION block (text + mark scheme) is a single atomic chunk
   → The question and its evaluation criteria are always retrieved together
```

### Why five specialised prompt templates?

Each of the 5 test questions is a different cognitive task:
- Syllabus mapping → requires AO cross-referencing
- Concept gap analysis → requires mark scheme level interpretation
- Mark weighting → requires quantitative comparison + inference
- Practice generation → requires style matching + constraint following
- Revision planning → requires prioritisation under time constraint

A single generic prompt handles none of these well. `classify_question()` routes to the
appropriate template deterministically — no LLM deciding which prompt to use.

---

## Architecture Diagram

```
   exam_paper.pdf / .txt
          │
          ▼
   ┌─────────────┐
   │  ingest.py  │   extract → parse metadata → chunk at question boundaries
   └──────┬──────┘
          │  chunks + metadata
          ▼
   ┌─────────────┐
   │  ChromaDB   │   pgvector-compatible local store
   │ (chroma_db) │   cosine similarity, persisted to disk
   └──────┬──────┘
          │
   ╔══════╧═══════════════════════════════════╗
   ║  query.py — RAG Engine (per question)   ║
   ║                                          ║
   ║  1. classify_question() → prompt type   ║
   ║  2. embed query (sentence-transformers) ║
   ║  3. retrieve top-k chunks (ChromaDB)    ║
   ║  4. assemble_context() (explicit)       ║
   ║  5. build_prompt() (board-aware)        ║
   ║  6. call_llm() → Ollama / Anthropic     ║
   ║  7. log result + return                 ║
   ╚══════════════════════════════════════════╝
          │
          ▼
   output/results.txt + results.json
```

---

## Known Limitations

1. **Single paper scope** — the vector store holds one exam paper. Production would index
   an entire corpus of past papers per board.

2. **No re-ranking** — retrieval is pure ANN cosine similarity. A cross-encoder re-ranker
   (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) would improve precision for ambiguous queries.

3. **Keyword-based query classifier** — `classify_question()` uses string matching.
   A small trained classifier would handle edge cases more robustly.

4. **No student profile** — answers are paper-level, not personalised. Production would
   inject the student's historical performance per topic into the system prompt.

5. **Mistral 7B reasoning depth** — for complex multi-step questions (e.g. practice question
   generation), GPT-4o or Claude Sonnet produces noticeably better output. Ollama is
   appropriate for this demo; the query.py backend switcher handles the upgrade path.

6. **No eval harness** — a production system would run ground-truth Q&A pairs nightly and
   alert on retrieval or answer quality regressions.

---

## What I Would Improve With More Time

**Retrieval quality:**
- Add a cross-encoder re-ranking step post-ANN retrieval
- Implement hybrid search (BM25 sparse + dense embeddings) for exact keyword matches
  like question numbers ("Q01", "Section B")
- Store embeddings by SHA-256 content hash to skip re-embedding identical content on re-ingest

**Prompt quality:**
- Add few-shot examples to each prompt template from the AQA standardisation materials
- Implement prompt versioning table (as described in the Part 2 strategic response)
- A/B test prompt variants against eval ground truth

**Architecture:**
- Replace ChromaDB with pgvector on PostgreSQL — co-locates with relational data,
  enables filtering by board_id/year/topic without a separate vector DB
- Add the Curriculum Knowledge Graph overlay (board-specific AO weightings injected
  into every system prompt from a database table, not hardcoded)
- Log every query and response to a signal store table for future fine-tuning data

**Observability:**
- Structured logging: every LLM call logged with tokens_in, tokens_out, latency_ms,
  retrieved_chunk_ids, query_type
- A results diff tool: compare answers across model versions or prompt versions

---

## Sample Output

See `output/results.txt` for pre-run answers on the AQA GCSE History paper included
in `data/exam_paper.txt`.

---

*Built for Acarnae technical assessment — Part 1.*
*Stack: Python · PyMuPDF · sentence-transformers · ChromaDB · Ollama (Mistral)*
*No API keys required. Runs offline.*
