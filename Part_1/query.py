"""
query.py — RAG query engine for the Acarnae pipeline.

For each incoming question:
  1. Embed the query (locally, no API)
  2. Retrieve top-k relevant chunks from ChromaDB
  3. Assemble explicit prompt (system + board context + chunks + question)
  4. Call Ollama (local LLM) — falls back to OpenAI if configured
  5. Log and return the response

No LangChain. No LlamaIndex. Every step is inspectable.
"""

import json
import os
import time
from pathlib import Path

import chromadb
import requests
from sentence_transformers import SentenceTransformer

from prompts import build_prompt, classify_question

# ─── Config ───────────────────────────────────────────────────────────────────
CHROMA_DIR    = "./chroma_db"
COLLECTION    = "exam_paper"
EMBED_MODEL   = "all-MiniLM-L6-v2"
OLLAMA_URL    = "http://localhost:11434/api/chat"
OLLAMA_MODEL  = "mistral"        # or "llama3.2", "phi3" — whatever is pulled
TOP_K         = 4                # chunks retrieved per query


# ─── LLM backends ─────────────────────────────────────────────────────────────

def call_ollama(system_prompt: str, user_prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Call local Ollama instance. Streams and collects response.
    Raises ConnectionError if Ollama is not running.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,   # lower = more deterministic for exam content
            "num_predict": 1200,
        }
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Ollama is not running. Start it with: ollama serve\n"
            "Then pull a model: ollama pull mistral"
        )


def call_openai(system_prompt: str, user_prompt: str) -> str:
    """
    Fallback: call OpenAI API if OPENAI_API_KEY is set.
    Direct HTTP — no SDK.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set and Ollama not available.")

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1200,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        json=payload, headers=headers, timeout=60
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def call_anthropic(system_prompt: str, user_prompt: str) -> str:
    """
    Fallback: call Anthropic Claude API if ANTHROPIC_API_KEY is set.
    Direct HTTP — no SDK.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set.")

    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1200,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        json=payload, headers=headers, timeout=60
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"].strip()


def call_llm(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    """
    Try LLM backends in priority order:
      1. Ollama (local, no cost, preferred)
      2. Anthropic Claude (if ANTHROPIC_API_KEY set)
      3. OpenAI (if OPENAI_API_KEY set)

    Returns (response_text, backend_used)
    """
    # Try Ollama first
    try:
        response = call_ollama(system_prompt, user_prompt)
        return response, "ollama"
    except ConnectionError:
        pass

    # Try Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            response = call_anthropic(system_prompt, user_prompt)
            return response, "anthropic"
        except Exception as e:
            print(f"  [warn] Anthropic call failed: {e}")

    # Try OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        try:
            response = call_openai(system_prompt, user_prompt)
            return response, "openai"
        except Exception as e:
            print(f"  [warn] OpenAI call failed: {e}")

    raise RuntimeError(
        "No LLM backend available.\n"
        "Option 1 (recommended): Install Ollama → https://ollama.com → run: ollama pull mistral\n"
        "Option 2: Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable"
    )


# ─── Retrieval ────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Minimal RAG engine. Explicit, inspectable, no framework magic.

    Lifecycle:
        engine = RAGEngine()
        result = engine.query("What topics does Section B cover?")
        print(result["answer"])
    """

    def __init__(self):
        # Load embedding model (cached after first download)
        print("  Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_collection(COLLECTION)

        # Load exam metadata
        meta_path = os.path.join(CHROMA_DIR, "exam_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.exam_meta = json.load(f)
        else:
            self.exam_meta = {"board": "AQA", "qualification": "GCSE History"}

        print(f"  Ready. Board: {self.exam_meta.get('board')} | "
              f"Paper: {self.exam_meta.get('paper_code', 'Unknown')}\n")

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        """
        Embed query → ANN search in ChromaDB → return top-k chunks.

        Returns list of dicts: {id, text, metadata, distance}
        """
        query_embedding = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i, doc_id in enumerate(results["ids"][0]):
            chunks.append({
                "id":       doc_id,
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return chunks

    def assemble_context(self, chunks: list[dict]) -> str:
        """
        Assemble retrieved chunks into a single context block.
        Ordered by relevance (closest distance first).
        Labels each chunk by type so the LLM understands the structure.
        """
        parts = []
        for chunk in sorted(chunks, key=lambda x: x["distance"]):
            ctype = chunk["metadata"].get("chunk_type", "content")
            label = f"[{ctype.upper().replace('_', ' ')}]"
            parts.append(f"{label}\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def query(self, question: str, verbose: bool = True) -> dict:
        """
        Full RAG pipeline for a single question.

        Returns dict with:
            question, query_type, retrieved_chunks, context,
            system_prompt, user_prompt, answer, backend, latency_ms
        """
        t_start = time.time()

        if verbose:
            print(f"  Query type: ", end="")

        # Step 1: Classify query → select specialised prompt template
        query_type = classify_question(question)
        if verbose:
            print(query_type)

        # Step 2: Retrieve relevant chunks
        if verbose:
            print(f"  Retrieving top-{TOP_K} chunks...", end="")
        chunks = self.retrieve(question, k=TOP_K)
        if verbose:
            ids = [c['id'] for c in chunks]
            print(f" retrieved: {ids}")

        # Step 3: Assemble context
        context = self.assemble_context(chunks)

        # Step 4: Build explicit prompt
        system_prompt, user_prompt, _ = build_prompt(
            question_text=question,
            context=context,
            board=self.exam_meta.get("board", "AQA"),
            qualification=self.exam_meta.get("qualification", "GCSE History"),
        )

        # Step 5: Call LLM
        if verbose:
            print(f"  Calling LLM...", end="", flush=True)
        answer, backend = call_llm(system_prompt, user_prompt)
        latency_ms = int((time.time() - t_start) * 1000)
        if verbose:
            print(f" done via {backend} ({latency_ms}ms)")

        return {
            "question":         question,
            "query_type":       query_type,
            "retrieved_chunks": [c["id"] for c in chunks],
            "context_length":   len(context),
            "answer":           answer,
            "backend":          backend,
            "latency_ms":       latency_ms,
        }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Acarnae RAG — Query engine")
    parser.add_argument("question", nargs="?",
                        default="What topics are covered in this exam paper?",
                        help="Question to ask about the exam paper")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Acarnae RAG — Query Engine")
    print(f"{'='*60}\n")

    engine = RAGEngine()
    result = engine.query(args.question, verbose=args.verbose)

    print(f"\n{'─'*60}")
    print(f"QUESTION:\n{result['question']}\n")
    print(f"ANSWER:\n{result['answer']}")
    print(f"{'─'*60}")
    print(f"[meta] type={result['query_type']} | "
          f"chunks={result['retrieved_chunks']} | "
          f"backend={result['backend']} | "
          f"latency={result['latency_ms']}ms")
