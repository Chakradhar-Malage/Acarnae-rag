"""
pipeline.py — End-to-end pipeline runner for the Acarnae RAG assessment.

Runs all 5 sample questions from the assessment brief against the ingested
exam paper and writes results to output/results.json and output/results.txt.

Usage:
    python pipeline.py                    # Run all 5 questions
    python pipeline.py --question 3       # Run question 3 only
    python pipeline.py --ingest           # Re-ingest before running
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── The 5 sample questions from the assessment brief ─────────────────────────
SAMPLE_QUESTIONS = [
    {
        "id": 1,
        "category": "Syllabus Mapping",
        "question": (
            "What topics from the syllabus are covered in Section A of this paper, "
            "and how do they map to the official curriculum objectives?"
        ),
    },
    {
        "id": 2,
        "category": "Concept Gap Analysis",
        "question": (
            "A student scored 0 on Question 01. Based on the marking scheme logic "
            "visible in this paper, what specific concept gaps does that likely indicate?"
        ),
    },
    {
        "id": 3,
        "category": "Mark Weighting & Examiner Priorities",
        "question": (
            "Which questions in this paper carry the highest mark weighting, "
            "and what does that suggest about examiners' topic priorities for this board?"
        ),
    },
    {
        "id": 4,
        "category": "Practice Question Generation",
        "question": (
            "Generate three practice questions in the same style and difficulty as "
            "Question 02, staying within the same syllabus topic of historical "
            "interpretations and provenance analysis."
        ),
    },
    {
        "id": 5,
        "category": "Revision Priority Planning",
        "question": (
            "A student has 3 weeks until this exam. Based on the content distribution "
            "in this paper, suggest a revision priority order across the topics tested."
        ),
    },
]


# ─── Utilities ────────────────────────────────────────────────────────────────

def ensure_ingested(file_path: str = "data/exam_paper.txt") -> None:
    """Check ChromaDB has data; if not, run ingestion automatically."""
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    collections = [c.name for c in client.list_collections()]

    if "exam_paper" not in collections:
        print("[pipeline] No ingested data found. Running ingest.py first...\n")
        from ingest import ingest
        ingest(file_path)
    else:
        coll = client.get_collection("exam_paper")
        count = coll.count()
        if count == 0:
            print("[pipeline] Collection empty. Re-ingesting...\n")
            from ingest import ingest
            ingest(file_path, reset=True)
        else:
            print(f"[pipeline] Found {count} chunks in vector store. Skipping ingest.\n")


def print_divider(char="═", width=64):
    print(char * width)


def format_result_block(result: dict, q_meta: dict) -> str:
    """Format a single Q&A result for the text output file."""
    lines = [
        f"{'═'*64}",
        f"  Q{q_meta['id']}: {q_meta['category']}",
        f"{'─'*64}",
        f"QUESTION:\n{q_meta['question']}",
        f"{'─'*64}",
        f"ANSWER:\n{result['answer']}",
        f"{'─'*64}",
        f"[meta] query_type={result['query_type']} | "
        f"chunks={result['retrieved_chunks']} | "
        f"backend={result['backend']} | "
        f"latency={result['latency_ms']}ms",
        "",
    ]
    return "\n".join(lines)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(question_ids: list[int] = None,
                 exam_file: str = "data/exam_paper.txt",
                 output_dir: str = "output") -> None:
    """
    Run the full RAG pipeline against selected questions.

    Args:
        question_ids: List of question numbers to run (1–5). None = all.
        exam_file:    Path to exam paper for ingestion if needed.
        output_dir:   Directory to write results.
    """
    print_divider()
    print("  Acarnae RAG Pipeline — Assessment Runner")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_divider()
    print()

    # 1. Ensure data is ingested
    ensure_ingested(exam_file)

    # 2. Load RAG engine
    print("[pipeline] Initialising RAG engine...")
    from query import RAGEngine
    engine = RAGEngine()

    # 3. Select questions
    questions = SAMPLE_QUESTIONS
    if question_ids:
        questions = [q for q in SAMPLE_QUESTIONS if q["id"] in question_ids]
        if not questions:
            print(f"No questions found for ids: {question_ids}")
            sys.exit(1)

    # 4. Run each question
    all_results = []
    total_start = time.time()

    for q_meta in questions:
        print_divider("─")
        print(f"  Running Q{q_meta['id']}: {q_meta['category']}")
        print_divider("─")

        result = engine.query(q_meta["question"], verbose=True)
        result["question_id"] = q_meta["id"]
        result["category"]    = q_meta["category"]
        all_results.append((q_meta, result))

        # Print answer immediately
        print(f"\nANSWER:\n{result['answer']}\n")

    total_elapsed = int((time.time() - total_start) * 1000)

    # 5. Write outputs
    os.makedirs(output_dir, exist_ok=True)

    # JSON output (machine-readable)
    json_path = os.path.join(output_dir, "results.json")
    json_data = {
        "run_at": datetime.now().isoformat(),
        "total_latency_ms": total_elapsed,
        "questions_run": len(all_results),
        "results": [
            {
                "id":       qm["id"],
                "category": qm["category"],
                "question": qm["question"],
                **r,
            }
            for qm, r in all_results
        ]
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # Text output (human-readable, for submission)
    txt_path = os.path.join(output_dir, "results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("ACARNAE RAG PIPELINE — SAMPLE OUTPUT\n")
        f.write(f"Run at: {datetime.now().isoformat()}\n")
        f.write(f"Exam paper: {exam_file}\n\n")
        for q_meta, result in all_results:
            f.write(format_result_block(result, q_meta) + "\n")
        f.write(f"\nTotal pipeline time: {total_elapsed}ms\n")

    print_divider()
    print(f"  Pipeline complete.")
    print(f"  Questions run : {len(all_results)}")
    print(f"  Total time    : {total_elapsed}ms")
    print(f"  Results saved : {txt_path}")
    print(f"               : {json_path}")
    print_divider()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acarnae RAG — Pipeline runner")
    parser.add_argument(
        "--question", type=int, nargs="*",
        help="Question number(s) to run (1–5). Default: all."
    )
    parser.add_argument(
        "--file", default="data/exam_paper.txt",
        help="Path to exam paper PDF or TXT"
    )
    parser.add_argument(
        "--ingest", action="store_true",
        help="Force re-ingestion before running"
    )
    parser.add_argument(
        "--output", default="output",
        help="Output directory for results"
    )
    args = parser.parse_args()

    if args.ingest:
        from ingest import ingest
        ingest(args.file, reset=True)

    run_pipeline(
        question_ids=args.question,
        exam_file=args.file,
        output_dir=args.output,
    )
