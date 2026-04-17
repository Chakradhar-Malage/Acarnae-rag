"""
ingest.py — PDF ingestion pipeline for the RAG system.

Parses an exam paper PDF (or .txt for demo), splits into structured chunks
at question boundaries, embeds with sentence-transformers, and stores in ChromaDB.

Run:
    python ingest.py --file data/exam_paper.txt
    python ingest.py --file your_exam_paper.pdf

No API keys required — embeddings run entirely locally.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ─── Config ──────
CHROMA_DIR   = "./chroma_db"
COLLECTION   = "exam_paper"
EMBED_MODEL  = "all-MiniLM-L6-v2"   # ~80MB, downloads once, runs fully locally


# ─── Text extraction ──────

def extract_text_from_pdf(path: str) -> str:
    """Extract raw text from a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF not installed. Run: pip install pymupdf")
        sys.exit(1)

    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n".join(pages)


def extract_text_from_txt(path: str) -> str:
    """Read plain text file (used for demo with pre-structured exam content)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text(path: str) -> str:
    """Route to correct extractor based on file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt")


# ─── Chunking ─────────────────────────────────────────────────────────────────

def parse_metadata_block(text: str) -> dict:
    """
    Extract structured metadata from the header of the exam file.
    Returns dict with board, qualification, paper_code, date, topic.
    """
    meta = {}
    patterns = {
        "board":           r"EXAM BOARD:\s*(.+)",
        "qualification":   r"QUALIFICATION:\s*(.+)",
        "paper_code":      r"PAPER CODE:\s*(.+)",
        "date":            r"DATE:\s*(.+)",
        "topic":           r"TOPIC:\s*(.+)",
        "section":         r"SECTION:\s*(.+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        meta[key] = m.group(1).strip() if m else "Unknown"
    return meta


def chunk_by_questions(text: str, metadata: dict) -> list[dict]:
    """
    Split exam paper text into semantic chunks.

    Chunking strategy (exam-aware, not naive paragraph splitting):
    1. Each QUESTION block becomes its own chunk (primary retrieval unit)
    2. Each INTERPRETATION becomes its own chunk (source material)
    3. General mark scheme instructions → single chunk
    4. Curriculum mapping → single chunk
    5. Metadata header → single chunk

    This preserves the question–mark-scheme relationship that generic
    chunkers destroy by splitting on token/paragraph boundaries.
    """
    chunks = []

    # ── Chunk 1: Interpretations ──────────────────────────────────
    interp_matches = re.finditer(
        r"(INTERPRETATION [A-Z].*?)(?=INTERPRETATION [A-Z]|={4,}|$)",
        text, re.DOTALL
    )
    for m in interp_matches:
        block = m.group(0).strip()
        if len(block) > 50:
            label = re.search(r"INTERPRETATION ([A-Z])", block)
            chunks.append({
                "id":      f"interpretation_{label.group(1) if label else 'X'}",
                "type":    "interpretation",
                "label":   label.group(1) if label else "Unknown",
                "text":    block,
                "metadata": {**metadata, "chunk_type": "interpretation"},
            })

    # ── Chunk 2: Questions + mark schemes ─────────────────────────
    # Match QUESTION blocks delimited by the structured format
    question_pattern = re.compile(
        r"QUESTION:\s*(\d+)(.*?)(?=QUESTION:\s*\d+|={4,}MARK SCHEME — GENERAL|={4,}CURRICULUM|$)",
        re.DOTALL
    )
    for m in question_pattern.finditer(text):
        q_num  = m.group(1).strip()
        q_body = m.group(2).strip()

        # Extract marks from block
        marks_m = re.search(r"Marks:\s*(\d+)", q_body)
        marks   = int(marks_m.group(1)) if marks_m else 0

        # Extract AOs
        ao_m  = re.search(r"Assessment Objective:\s*(.+)", q_body)
        aos   = ao_m.group(1).strip() if ao_m else ""

        # Extract question text
        qt_m  = re.search(r"Question Text:\s*(.+?)(?=MARK SCHEME|$)", q_body, re.DOTALL)
        q_text = qt_m.group(1).strip() if qt_m else q_body[:300]

        chunks.append({
            "id":    f"question_{q_num}",
            "type":  "question",
            "question_number": q_num,
            "marks": marks,
            "assessment_objectives": aos,
            "question_text": q_text,
            "text":  f"QUESTION {q_num} [{marks} marks]\n{q_body}",
            "metadata": {
                **metadata,
                "chunk_type": "question",
                "question_number": q_num,
                "marks": str(marks),
                "assessment_objectives": aos,
            },
        })

    # ── Chunk 3: General mark scheme instructions ─────────────────
    ms_m = re.search(
        r"(MARK SCHEME — GENERAL INSTRUCTIONS.*?)(?=={4,}CURRICULUM|$)",
        text, re.DOTALL
    )
    if ms_m:
        chunks.append({
            "id":   "mark_scheme_general",
            "type": "mark_scheme_instructions",
            "text": ms_m.group(1).strip(),
            "metadata": {**metadata, "chunk_type": "mark_scheme_instructions"},
        })

    # ── Chunk 4: Curriculum mapping ───────────────────────────────
    curr_m = re.search(
        r"(CURRICULUM MAPPING.*?)$",
        text, re.DOTALL
    )
    if curr_m:
        chunks.append({
            "id":   "curriculum_mapping",
            "type": "curriculum_mapping",
            "text": curr_m.group(1).strip(),
            "metadata": {**metadata, "chunk_type": "curriculum_mapping"},
        })

    # ── Chunk 5: Full document (for broad queries) ────────────────
    chunks.append({
        "id":   "full_document",
        "type": "full_document",
        "text": text[:6000],  # first 6000 chars as a broad-context chunk
        "metadata": {**metadata, "chunk_type": "full_document"},
    })

    return chunks


# ─── Embedding & storage ──────────────────────────────────────────────────────

def content_hash(text: str) -> str:
    """SHA-256 hash of chunk text — used to skip re-embedding identical content."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def ingest(file_path: str, reset: bool = False) -> None:
    """
    Full ingestion pipeline:
      extract → parse metadata → chunk → embed → store in ChromaDB

    Args:
        file_path: Path to PDF or TXT exam paper
        reset:     If True, wipe and rebuild the collection from scratch
    """
    print(f"\n{'='*60}")
    print(f"  Acarnae RAG — Ingestion Pipeline")
    print(f"{'='*60}")
    print(f"  File:   {file_path}")
    print(f"  Model:  {EMBED_MODEL} (local, no API key)")
    print(f"  Store:  {CHROMA_DIR}")
    print(f"{'='*60}\n")

    # 1. Extract text
    print("[1/4] Extracting text from document...")
    text = extract_text(file_path)
    print(f"      Extracted {len(text):,} characters\n")

    # 2. Parse metadata + chunk
    print("[2/4] Parsing structure and chunking...")
    metadata = parse_metadata_block(text)
    print(f"      Board:         {metadata['board']}")
    print(f"      Qualification: {metadata['qualification']}")
    print(f"      Paper:         {metadata['paper_code']}")
    print(f"      Topic:         {metadata['topic']}")

    chunks = chunk_by_questions(text, metadata)
    print(f"\n      Created {len(chunks)} chunks:")
    for c in chunks:
        ctype = c['type']
        cid   = c['id']
        words = len(c['text'].split())
        print(f"        [{ctype:30s}] {cid:25s} ({words} words)")
    print()

    # 3. Embed locally
    print(f"[3/4] Embedding chunks with {EMBED_MODEL}...")
    print(f"      (First run downloads ~80MB model — cached after that)")
    model  = SentenceTransformer(EMBED_MODEL)
    texts  = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    print(f"      Embedded {len(embeddings)} chunks, dimension={embeddings.shape[1]}\n")

    # 4. Store in ChromaDB
    print("[4/4] Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if reset and COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
        print(f"      Wiped existing collection '{COLLECTION}'")

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert — safe to re-run, deduplicates by id
    ids        = [c["id"] for c in chunks]
    metadatas  = [c["metadata"] for c in chunks]
    documents  = [c["text"] for c in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=documents,
    )

    # Save metadata for query.py to read
    meta_path = os.path.join(CHROMA_DIR, "exam_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"      Stored {len(chunks)} chunks in collection '{COLLECTION}'")
    print(f"      Metadata saved to {meta_path}")
    print(f"\n{'='*60}")
    print(f"  Ingestion complete. Run pipeline.py to query.")
    print(f"{'='*60}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acarnae RAG — Ingest exam paper")
    parser.add_argument("--file",  default="data/exam_paper.txt",
                        help="Path to exam paper PDF or TXT")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe and rebuild the vector store from scratch")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    ingest(args.file, reset=args.reset)
