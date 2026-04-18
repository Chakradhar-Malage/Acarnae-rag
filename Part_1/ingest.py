"""
ingest.py — PDF ingestion pipeline for the Acarnae RAG system.

Handles two input formats:
  1. Real AQA exam PDFs (auto-detected, layout-aware chunking)
  2. Structured .txt demo files

Run:
    python ingest.py --file data/exam_paper.PDF --reset
    python ingest.py --file data/exam_paper.txt --reset
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR  = "./chroma_db"
COLLECTION  = "exam_paper"
EMBED_MODEL = "all-MiniLM-L6-v2"


# ─── Text extraction ──────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    try:
        import fitz
    except ImportError:
        print("PyMuPDF not installed. Run: pip install pymupdf")
        sys.exit(1)
    doc = fitz.open(path)
    return "\n".join(page.get_text("text") for page in doc)

def extract_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ─── Metadata detection ───────────────────────────────────────────

def detect_meta(text: str) -> dict:
    meta = {"board": "Unknown", "qualification": "Unknown",
            "paper_code": "Unknown", "date": "Unknown", "topic": "Unknown"}

    # Our structured format
    for key, pat in [
        ("board",         r"EXAM BOARD:\s*(.+)"),
        ("qualification", r"QUALIFICATION:\s*(.+)"),
        ("paper_code",    r"PAPER CODE:\s*(.+)"),
        ("date",          r"DATE:\s*(.+)"),
        ("topic",         r"TOPIC:\s*(.+)"),
    ]:
        m = re.search(pat, text)
        if m:
            meta[key] = m.group(1).strip()

    # Real PDF fallback detection
    if meta["board"] == "Unknown":
        for board in ["AQA", "Edexcel", "OCR", "Cambridge", "CAIE", "WJEC"]:
            if board in text[:600]:
                meta["board"] = board
                break

    if meta["qualification"] == "Unknown":
        for qual in ["GCSE", "A Level", "AS Level", "IGCSE"]:
            if qual in text[:1000]:
                meta["qualification"] = qual
                break

    if meta["paper_code"] == "Unknown":
        m = re.search(r"\b(\d{4}/\d[A-Z]?/[A-Z0-9]+)\b", text)
        if m:
            meta["paper_code"] = m.group(1)

    if meta["date"] == "Unknown":
        m = re.search(r"(June|November|January|May)\s+(20\d{2})", text, re.IGNORECASE)
        if m:
            meta["date"] = m.group(0)

    if meta["topic"] == "Unknown":
        m = re.search(r"(?:GCSE|A\s*Level)\s+([A-Z][A-Z\s]{3,30})(?:\s*[–\-]|\s*\n)", text[:800])
        if m:
            meta["topic"] = m.group(0).strip()[:60]

    return meta


# ─── Chunking ─────────────────────────────────────────────────────

def is_structured_txt(text: str) -> bool:
    return "EXAM BOARD:" in text and "QUESTION:" in text


def chunk_structured_txt(text: str, meta: dict) -> list[dict]:
    """Chunker for our structured .txt demo format."""
    chunks = []

    for m in re.finditer(
        r"(INTERPRETATION [A-Z].*?)(?=INTERPRETATION [A-Z]|={4,}|$)", text, re.DOTALL
    ):
        block = m.group(0).strip()
        if len(block) > 50:
            lbl = re.search(r"INTERPRETATION ([A-Z])", block)
            chunks.append({
                "id": f"interpretation_{lbl.group(1) if lbl else 'X'}",
                "type": "interpretation", "text": block,
                "metadata": {**meta, "chunk_type": "interpretation"},
            })

    for m in re.compile(
        r"QUESTION:\s*(\d+)(.*?)(?=QUESTION:\s*\d+|={4,}MARK SCHEME — GENERAL|={4,}CURRICULUM|$)",
        re.DOTALL
    ).finditer(text):
        q_num = m.group(1).strip()
        body  = m.group(2).strip()
        marks = int(mm.group(1)) if (mm := re.search(r"Marks:\s*(\d+)", body)) else 0
        aos   = am.group(1).strip() if (am := re.search(r"Assessment Objective:\s*(.+)", body)) else ""
        chunks.append({
            "id": f"question_{q_num}", "type": "question",
            "text": f"QUESTION {q_num} [{marks} marks]\n{body}",
            "metadata": {**meta, "chunk_type": "question", "question_number": q_num,
                         "marks": str(marks), "assessment_objectives": aos},
        })

    ms = re.search(r"(MARK SCHEME — GENERAL INSTRUCTIONS.*?)(?=={4,}CURRICULUM|$)", text, re.DOTALL)
    if ms:
        chunks.append({"id": "mark_scheme_general", "type": "mark_scheme_instructions",
                       "text": ms.group(1).strip(),
                       "metadata": {**meta, "chunk_type": "mark_scheme_instructions"}})

    cm = re.search(r"(CURRICULUM MAPPING.*?)$", text, re.DOTALL)
    if cm:
        chunks.append({"id": "curriculum_mapping", "type": "curriculum_mapping",
                       "text": cm.group(1).strip(),
                       "metadata": {**meta, "chunk_type": "curriculum_mapping"}})

    chunks.append({"id": "full_document", "type": "full_document",
                   "text": text[:6000], "metadata": {**meta, "chunk_type": "full_document"}})
    return chunks


def chunk_real_pdf(text: str, meta: dict) -> list[dict]:
    """
    Chunker for real AQA exam PDFs.

    Strategy:
    - Detect mark scheme boundary and split paper into
      question_paper section + mark_scheme section
    - Extract interpretation/source blocks as individual chunks
    - Extract each AQA question (0 1, 0 2 ...) as a chunk
    - Merge each question with its corresponding mark scheme
    - Add sliding-window context chunks for broad queries
    """
    chunks = []
    clean  = re.sub(r"\n{3,}", "\n\n", text)

    # ── Find mark scheme boundary ─────────────────────────────────
    ms_header = re.search(
        r"MARK SCHEME\s*[–\-].*?(?=\n)", clean, re.IGNORECASE
    )
    if ms_header:
        q_text = clean[:ms_header.start()]
        ms_text = clean[ms_header.start():]
    else:
        q_text  = clean
        ms_text = ""

    # ── Interpretation blocks ─────────────────────────────────────
    for m in re.finditer(
        r"(Interpretation\s+[A-Z][\s\S]{100,1500}?)(?=Interpretation\s+[A-Z]|END OF INTERP|ANSWER ALL|\Z)",
        clean, re.IGNORECASE
    ):
        block = m.group(0).strip()
        lbl_m = re.search(r"Interpretation\s+([A-Z])", block, re.IGNORECASE)
        lbl   = lbl_m.group(1).upper() if lbl_m else "X"
        chunks.append({
            "id": f"interpretation_{lbl}", "type": "interpretation",
            "text": f"[SOURCE: Interpretation {lbl}]\n{block}",
            "metadata": {**meta, "chunk_type": "interpretation", "label": lbl},
        })

    # ── Question chunks from question paper section ───────────────
    # AQA format: questions start with "0 1", "0 2" on their own line
    q_blocks = re.split(r"(?=\n\s*0\s+[1-9]\b)", q_text)
    for seg in q_blocks:
        seg = seg.strip()
        if not seg or len(seg) < 60:
            continue
        qm = re.match(r"0\s+([1-9])", seg)
        if not qm:
            continue
        q_num   = qm.group(1)
        marks_m = re.search(r"\[(\d+)\s+marks?\]", seg, re.IGNORECASE)
        marks   = int(marks_m.group(1)) if marks_m else 0
        chunks.append({
            "id": f"question_{q_num}", "type": "question",
            "text": f"[QUESTION 0{q_num}] [{marks} marks]\n{seg}",
            "metadata": {**meta, "chunk_type": "question",
                         "question_number": q_num, "marks": str(marks)},
        })

    # ── Mark scheme blocks — merge into question chunks ───────────
    if ms_text:
        # General instructions
        gen_m = re.search(
            r"(Level of response marking instructions[\s\S]{0,2000}?)(?=\n\s*0\s+[1-9]|\Z)",
            ms_text, re.IGNORECASE
        )
        if gen_m:
            chunks.append({
                "id": "mark_scheme_general", "type": "mark_scheme_instructions",
                "text": f"[GENERAL MARKING INSTRUCTIONS]\n{gen_m.group(1).strip()}",
                "metadata": {**meta, "chunk_type": "mark_scheme_instructions"},
            })

        # Per-question mark schemes
        ms_blocks = re.split(r"(?=\n\s*0\s+[1-9]\b)", ms_text)
        for seg in ms_blocks:
            seg = seg.strip()
            if not seg or len(seg) < 60:
                continue
            qm = re.match(r"0\s+([1-9])", seg)
            if not qm:
                continue
            q_num = qm.group(1)
            # Try to merge with existing question chunk
            existing = next((c for c in chunks if c["id"] == f"question_{q_num}"), None)
            if existing:
                existing["text"] += f"\n\n[MARK SCHEME FOR QUESTION 0{q_num}]\n{seg}"
                existing["id"]   = f"question_{q_num}_with_ms"
            else:
                chunks.append({
                    "id": f"mark_scheme_q{q_num}", "type": "mark_scheme",
                    "text": f"[MARK SCHEME Q0{q_num}]\n{seg}",
                    "metadata": {**meta, "chunk_type": "mark_scheme",
                                 "question_number": q_num},
                })

    # ── Sliding-window context chunks (for broad/summary queries) ─
    words = clean.split()
    win, step = 700, 500
    for i, start in enumerate(range(0, max(1, len(words) - win + 1), step)):
        seg = " ".join(words[start: start + win])
        if len(seg) < 100:
            continue
        cid = f"context_window_{i}"
        # Skip if this window is already fully covered by a specific chunk
        if not any(seg[:80] in c["text"] for c in chunks):
            chunks.append({
                "id": cid, "type": "context_window",
                "text": seg,
                "metadata": {**meta, "chunk_type": "context_window",
                             "window_index": str(i)},
            })

    # ── Full document fallback ─────────────────────────────────────
    chunks.append({
        "id": "full_document", "type": "full_document",
        "text": clean[:6000],
        "metadata": {**meta, "chunk_type": "full_document"},
    })

    return chunks


def chunk_document(text: str, meta: dict) -> list[dict]:
    if is_structured_txt(text):
        print("      Detected: structured .txt format")
        return chunk_structured_txt(text, meta)
    else:
        print("      Detected: real exam PDF — using layout-aware chunker")
        return chunk_real_pdf(text, meta)


# ─── Main ingestion ───────────────────────────────────────────────

def ingest(file_path: str, reset: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"  Acarnae RAG — Ingestion Pipeline")
    print(f"{'='*60}")
    print(f"  File:  {file_path}")
    print(f"  Model: {EMBED_MODEL} (local, no API key)")
    print(f"{'='*60}\n")

    print("[1/4] Extracting text...")
    text = extract_text(file_path)
    print(f"      Extracted {len(text):,} characters\n")

    print("[2/4] Detecting structure and chunking...")
    meta   = detect_meta(text)
    print(f"      Board:         {meta['board']}")
    print(f"      Qualification: {meta['qualification']}")
    print(f"      Paper:         {meta['paper_code']}")
    print(f"      Date:          {meta['date']}")
    print(f"      Topic:         {meta['topic']}")

    chunks = chunk_document(text, meta)
    print(f"\n      Created {len(chunks)} chunks:")
    for c in chunks:
        words = len(c["text"].split())
        print(f"        [{c['type']:28s}] {c['id']:38s} ({words:4d} words)")
    print()

    print(f"[3/4] Embedding with {EMBED_MODEL}...")
    model      = SentenceTransformer(EMBED_MODEL)
    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    print(f"      Embedded {len(embeddings)} chunks, dim={embeddings.shape[1]}\n")

    print("[4/4] Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if reset:
        try:
            client.delete_collection(COLLECTION)
            print("      Wiped existing collection")
        except Exception:
            pass
    coll = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    coll.upsert(
        ids        = [c["id"] for c in chunks],
        embeddings = embeddings.tolist(),
        metadatas  = [c["metadata"] for c in chunks],
        documents  = [c["text"] for c in chunks],
    )
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(os.path.join(CHROMA_DIR, "exam_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"      Stored {len(chunks)} chunks in '{COLLECTION}'")
    print(f"\n{'='*60}")
    print(f"  Done! Run: python pipeline.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",  default="data/exam_paper.txt")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    ingest(args.file, reset=args.reset)
