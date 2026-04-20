# """
# query.py — RAG query engine for the Acarnae pipeline.

# LLM backend priority:
#   1. Ollama (local, free) — install: https://ollama.com → ollama pull mistral
#   2. Anthropic Claude    — set ANTHROPIC_API_KEY env variable
#   3. OpenAI GPT-4o-mini  — set OPENAI_API_KEY env variable

# No LangChain. No LlamaIndex. Every step is explicit and inspectable.
# """

# import json
# import os
# import sys
# import time
# from pathlib import Path

# import chromadb
# import requests
# from sentence_transformers import SentenceTransformer

# from prompts import build_prompt, classify_question

# CHROMA_DIR   = "./chroma_db"
# COLLECTION   = "exam_paper"
# EMBED_MODEL  = "all-MiniLM-L6-v2"
# OLLAMA_URL   = "http://localhost:11434/api/chat"
# OLLAMA_MODEL = "mistral"
# TOP_K        = 4


# # ─── LLM backends ─────────────

# def check_ollama_running() -> bool:
#     """Quick check if Ollama server is up — does not block."""
#     try:
#         r = requests.get("http://localhost:11434/api/tags", timeout=3)
#         return r.status_code == 200
#     except Exception:
#         return False


# def check_ollama_model(model: str) -> bool:
#     """Check if the requested model is already pulled."""
#     try:
#         r = requests.get("http://localhost:11434/api/tags", timeout=3)
#         if r.status_code == 200:
#             models = [m["name"] for m in r.json().get("models", [])]
#             return any(model in m for m in models)
#     except Exception:
#         pass
#     return False


# def call_ollama(system_prompt: str, user_prompt: str, model: str = OLLAMA_MODEL) -> str:
#     payload = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user",   "content": user_prompt},
#         ],
#         "stream": False,
#         "options": {"temperature": 0.3, "num_predict": 1200},
#     }
#     resp = requests.post(OLLAMA_URL, json=payload, timeout=900)
#     resp.raise_for_status()
#     return resp.json()["message"]["content"].strip()


# def call_anthropic(system_prompt: str, user_prompt: str) -> str:
#     api_key = os.environ.get("ANTHROPIC_API_KEY")
#     if not api_key:
#         raise EnvironmentError("ANTHROPIC_API_KEY not set")
#     resp = requests.post(
#         "https://api.anthropic.com/v1/messages",
#         headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
#                  "Content-Type": "application/json"},
#         json={"model": "claude-haiku-4-5-20251001", "max_tokens": 1200,
#               "system": system_prompt,
#               "messages": [{"role": "user", "content": user_prompt}]},
#         timeout=60,
#     )
#     resp.raise_for_status()
#     return resp.json()["content"][0]["text"].strip()


# def call_openai(system_prompt: str, user_prompt: str) -> str:
#     api_key = os.environ.get("OPENAI_API_KEY")
#     if not api_key:
#         raise EnvironmentError("OPENAI_API_KEY not set")
#     resp = requests.post(
#         "https://api.openai.com/v1/chat/completions",
#         headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
#         json={"model": "gpt-4o-mini", "temperature": 0.3, "max_tokens": 1200,
#               "messages": [{"role": "system", "content": system_prompt},
#                            {"role": "user",   "content": user_prompt}]},
#         timeout=60,
#     )
#     resp.raise_for_status()
#     return resp.json()["choices"][0]["message"]["content"].strip()


# def call_llm(system_prompt: str, user_prompt: str) -> tuple:
#     """
#     Try backends in order: Ollama → Anthropic → OpenAI.
#     Prints clear setup instructions if nothing is available.
#     Returns (response_text, backend_name).
#     """
#     # 1. Try Ollama
#     if check_ollama_running():
#         if not check_ollama_model(OLLAMA_MODEL):
#             print(f"\n  [!] Ollama is running but model '{OLLAMA_MODEL}' is not pulled.")
#             print(f"      Run this in a separate terminal: ollama pull {OLLAMA_MODEL}")
#             print(f"      Then re-run pipeline.py\n")
#         else:
#             try:
#                 return call_ollama(system_prompt, user_prompt), "ollama"
#             except Exception as e:
#                 print(f"  [warn] Ollama call failed: {e}")
#     else:
#         if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
#             print("\n" + "="*60)
#             print("  NO LLM BACKEND FOUND")
#             print("="*60)
#             print("\n  OPTION 1 — Local (free, recommended):")
#             print("    1. Download Ollama: https://ollama.com/download")
#             print("    2. Open a NEW terminal and run: ollama serve")
#             print(f"    3. In another terminal run:    ollama pull {OLLAMA_MODEL}")
#             print("    4. Come back and re-run:        python pipeline.py\n")
#             print("  OPTION 2 — API key:")
#             print("    Windows PowerShell:")
#             print("      $env:ANTHROPIC_API_KEY='sk-ant-...'")
#             print("      $env:OPENAI_API_KEY='sk-...'")
#             print("    Git Bash / Mac / Linux:")
#             print("      export ANTHROPIC_API_KEY=sk-ant-...")
#             print("      export OPENAI_API_KEY=sk-...")
#             print("="*60 + "\n")
#             sys.exit(1)

#     # 2. Try Anthropic
#     if os.environ.get("ANTHROPIC_API_KEY"):
#         try:
#             return call_anthropic(system_prompt, user_prompt), "anthropic"
#         except Exception as e:
#             print(f"  [warn] Anthropic failed: {e}")

#     # 3. Try OpenAI
#     if os.environ.get("OPENAI_API_KEY"):
#         try:
#             return call_openai(system_prompt, user_prompt), "openai"
#         except Exception as e:
#             print(f"  [warn] OpenAI failed: {e}")

#     print("\n  [error] All LLM backends failed. See setup instructions above.")
#     sys.exit(1)


# # ─── RAG Engine ───────────────────────────────────────────────────

# class RAGEngine:
#     def __init__(self):
#         print("  Loading embedding model...")
#         self.embed_model = SentenceTransformer(EMBED_MODEL)
#         self.client      = chromadb.PersistentClient(path=CHROMA_DIR)

#         try:
#             self.collection = self.client.get_collection(COLLECTION)
#         except Exception:
#             print(f"\n  [error] No ingested data found in '{CHROMA_DIR}'.")
#             print(f"  Run first: python ingest.py --file data/exam_paper.PDF --reset\n")
#             sys.exit(1)

#         meta_path = os.path.join(CHROMA_DIR, "exam_meta.json")
#         self.exam_meta = {}
#         if os.path.exists(meta_path):
#             with open(meta_path) as f:
#                 self.exam_meta = json.load(f)

#         board = self.exam_meta.get("board", "Unknown")
#         paper = self.exam_meta.get("paper_code", "Unknown")
#         count = self.collection.count()
#         print(f"  Ready. Board: {board} | Paper: {paper} | Chunks: {count}\n")

#     def retrieve(self, query: str, k: int = TOP_K) -> list:
#         emb = self.embed_model.encode([query], normalize_embeddings=True).tolist()
#         res = self.collection.query(
#             query_embeddings=emb, n_results=min(k, self.collection.count()),
#             include=["documents", "metadatas", "distances"],
#         )
#         return [
#             {"id": doc_id, "text": res["documents"][0][i],
#              "metadata": res["metadatas"][0][i], "distance": res["distances"][0][i]}
#             for i, doc_id in enumerate(res["ids"][0])
#         ]

#     def assemble_context(self, chunks: list) -> str:
#         parts = []
#         for c in sorted(chunks, key=lambda x: x["distance"]):
#             ctype = c["metadata"].get("chunk_type", "content")
#             parts.append(f"[{ctype.upper().replace('_', ' ')}]\n{c['text']}")
#         return "\n\n---\n\n".join(parts)

#     def query(self, question: str, verbose: bool = True) -> dict:
#         t0 = time.time()

#         query_type = classify_question(question)
#         if verbose:
#             print(f"  Query type: {query_type}")

#         chunks = self.retrieve(question, k=TOP_K)
#         if verbose:
#             print(f"  Retrieved:  {[c['id'] for c in chunks]}")

#         context      = self.assemble_context(chunks)
#         sys_p, usr_p, _ = build_prompt(
#             question_text=question,
#             context=context,
#             board=self.exam_meta.get("board", "AQA"),
#             qualification=self.exam_meta.get("qualification", "GCSE History"),
#         )

#         if verbose:
#             print(f"  Calling LLM...", end="", flush=True)
#         answer, backend = call_llm(sys_p, usr_p)
#         latency = int((time.time() - t0) * 1000)
#         if verbose:
#             print(f" done via {backend} ({latency}ms)")

#         return {
#             "question":         question,
#             "query_type":       query_type,
#             "retrieved_chunks": [c["id"] for c in chunks],
#             "context_length":   len(context),
#             "answer":           answer,
#             "backend":          backend,
#             "latency_ms":       latency,
#         }


# # ─── CLI ──────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("question", nargs="?",
#                         default="What topics are covered in this exam paper?")
#     args = parser.parse_args()

#     print(f"\n{'='*60}\n  Acarnae RAG — Query Engine\n{'='*60}\n")
#     engine = RAGEngine()
#     result = engine.query(args.question)
#     print(f"\n{'─'*60}")
#     print(f"QUESTION:\n{result['question']}\n")
#     print(f"ANSWER:\n{result['answer']}")
#     print(f"{'─'*60}")
#     print(f"[meta] type={result['query_type']} | chunks={result['retrieved_chunks']} "
#           f"| backend={result['backend']} | latency={result['latency_ms']}ms")


"""
query.py — RAG query engine for the Acarnae pipeline.

LLM backend priority:
  1. Ollama (local, free) — install: https://ollama.com → ollama pull mistral
  2. Anthropic Claude    — set ANTHROPIC_API_KEY env variable
  3. OpenAI GPT-4o-mini  — set OPENAI_API_KEY env variable

No LangChain. No LlamaIndex. Every step is explicit and inspectable.
"""

import json
import os
import sys
import time
from pathlib import Path

import chromadb
import requests
from sentence_transformers import SentenceTransformer

from prompts import build_prompt, classify_question

CHROMA_DIR   = "./chroma_db"
COLLECTION   = "exam_paper"
EMBED_MODEL  = "all-MiniLM-L6-v2"
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral"
TOP_K        = 2


# ─── LLM backends ─────────────────────────────────────────────────

def check_ollama_running() -> bool:
    """Quick check if Ollama server is up — does not block."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def check_ollama_model(model: str) -> bool:
    """Check if the requested model is already pulled."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return any(model in m for m in models)
    except Exception:
        pass
    return False


def call_ollama(system_prompt: str, user_prompt: str, model: str = OLLAMA_MODEL) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 500},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def call_anthropic(system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"},
        json={"model": "claude-haiku-4-5-20251001", "max_tokens": 1200,
              "system": system_prompt,
              "messages": [{"role": "user", "content": user_prompt}]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"].strip()


def call_openai(system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "gpt-4o-mini", "temperature": 0.3, "max_tokens": 1200,
              "messages": [{"role": "system", "content": system_prompt},
                           {"role": "user",   "content": user_prompt}]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def call_llm(system_prompt: str, user_prompt: str) -> tuple:
    """
    Try backends in order: Ollama → Anthropic → OpenAI.
    Prints clear setup instructions if nothing is available.
    Returns (response_text, backend_name).
    """
    # 1. Try Ollama
    if check_ollama_running():
        if not check_ollama_model(OLLAMA_MODEL):
            print(f"\n  [!] Ollama is running but model '{OLLAMA_MODEL}' is not pulled.")
            print(f"      Run this in a separate terminal: ollama pull {OLLAMA_MODEL}")
            print(f"      Then re-run pipeline.py\n")
        else:
            try:
                return call_ollama(system_prompt, user_prompt), "ollama"
            except Exception as e:
                print(f"  [warn] Ollama call failed: {e}")
    else:
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            print("\n" + "="*60)
            print("  NO LLM BACKEND FOUND")
            print("="*60)
            print("\n  OPTION 1 — Local (free, recommended):")
            print("    1. Download Ollama: https://ollama.com/download")
            print("    2. Open a NEW terminal and run: ollama serve")
            print(f"    3. In another terminal run:    ollama pull {OLLAMA_MODEL}")
            print("    4. Come back and re-run:        python pipeline.py\n")
            print("  OPTION 2 — API key:")
            print("    Windows PowerShell:")
            print("      $env:ANTHROPIC_API_KEY='sk-ant-...'")
            print("      $env:OPENAI_API_KEY='sk-...'")
            print("    Git Bash / Mac / Linux:")
            print("      export ANTHROPIC_API_KEY=sk-ant-...")
            print("      export OPENAI_API_KEY=sk-...")
            print("="*60 + "\n")
            sys.exit(1)

    # 2. Try Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return call_anthropic(system_prompt, user_prompt), "anthropic"
        except Exception as e:
            print(f"  [warn] Anthropic failed: {e}")

    # 3. Try OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return call_openai(system_prompt, user_prompt), "openai"
        except Exception as e:
            print(f"  [warn] OpenAI failed: {e}")

    print("\n  [error] All LLM backends failed. See setup instructions above.")
    sys.exit(1)


# ─── RAG Engine ───────────────────────────────────────────────────

class RAGEngine:
    def __init__(self):
        print("  Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        self.client      = chromadb.PersistentClient(path=CHROMA_DIR)

        try:
            self.collection = self.client.get_collection(COLLECTION)
        except Exception:
            print(f"\n  [error] No ingested data found in '{CHROMA_DIR}'.")
            print(f"  Run first: python ingest.py --file data/exam_paper.PDF --reset\n")
            sys.exit(1)

        meta_path = os.path.join(CHROMA_DIR, "exam_meta.json")
        self.exam_meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.exam_meta = json.load(f)

        board = self.exam_meta.get("board", "Unknown")
        paper = self.exam_meta.get("paper_code", "Unknown")
        count = self.collection.count()
        print(f"  Ready. Board: {board} | Paper: {paper} | Chunks: {count}\n")

    def retrieve(self, query: str, k: int = TOP_K) -> list:
        emb = self.embed_model.encode([query], normalize_embeddings=True).tolist()
        res = self.collection.query(
            query_embeddings=emb, n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"id": doc_id, "text": res["documents"][0][i],
             "metadata": res["metadatas"][0][i], "distance": res["distances"][0][i]}
            for i, doc_id in enumerate(res["ids"][0])
        ]

    def assemble_context(self, chunks: list) -> str:
        parts = []
        for c in sorted(chunks, key=lambda x: x["distance"]):
            ctype = c["metadata"].get("chunk_type", "content")
            parts.append(f"[{ctype.upper().replace('_', ' ')}]\n{c['text']}")
        context = "\n\n---\n\n".join(parts)
        return context[:3000]  # Cap context for CPU inference speed

    def query(self, question: str, verbose: bool = True) -> dict:
        t0 = time.time()

        query_type = classify_question(question)
        if verbose:
            print(f"  Query type: {query_type}")

        chunks = self.retrieve(question, k=TOP_K)
        if verbose:
            print(f"  Retrieved:  {[c['id'] for c in chunks]}")

        context      = self.assemble_context(chunks)
        sys_p, usr_p, _ = build_prompt(
            question_text=question,
            context=context,
            board=self.exam_meta.get("board", "AQA"),
            qualification=self.exam_meta.get("qualification", "GCSE History"),
        )

        if verbose:
            print(f"  Calling LLM...", end="", flush=True)
        answer, backend = call_llm(sys_p, usr_p)
        latency = int((time.time() - t0) * 1000)
        if verbose:
            print(f" done via {backend} ({latency}ms)")

        return {
            "question":         question,
            "query_type":       query_type,
            "retrieved_chunks": [c["id"] for c in chunks],
            "context_length":   len(context),
            "answer":           answer,
            "backend":          backend,
            "latency_ms":       latency,
        }


# ─── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?",
                        default="What topics are covered in this exam paper?")
    args = parser.parse_args()

    print(f"\n{'='*60}\n  Acarnae RAG — Query Engine\n{'='*60}\n")
    engine = RAGEngine()
    result = engine.query(args.question)
    print(f"\n{'─'*60}")
    print(f"QUESTION:\n{result['question']}\n")
    print(f"ANSWER:\n{result['answer']}")
    print(f"{'─'*60}")
    print(f"[meta] type={result['query_type']} | chunks={result['retrieved_chunks']} "
          f"| backend={result['backend']} | latency={result['latency_ms']}ms")
