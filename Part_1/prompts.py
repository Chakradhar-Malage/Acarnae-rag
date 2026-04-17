"""
prompts.py — All prompt templates for the Acarnae RAG pipeline.

Prompts are explicit, versioned, and board-aware.
No wrapper framework decides what gets sent to the model — we own every token.
"""

PROMPT_VERSION = "1.0.0"

# ─── System prompt base ───────────────────────────────────────────────────────
SYSTEM_BASE = """You are an expert AI study assistant for Acarnae, specialising in \
{board} {qualification} examinations.

You have deep knowledge of:
- The {board} mark scheme conventions for {qualification}
- Assessment Objective weightings (AO1–AO4 as applicable)
- Level-of-response marking methodology used by {board} examiners
- The specific syllabus content for this paper

When answering, always:
1. Reference specific mark scheme levels where relevant
2. Use precise curriculum language (Assessment Objectives, syllabus topic names)
3. Be specific — generic advice does not help students improve their grade
4. Cite evidence from the exam paper content provided

Exam paper context:
{context}
"""

# ─── Query-type prompt templates ─────────────────────────────────────────────

# Q1 type: Syllabus topic mapping
SYLLABUS_MAPPING_PROMPT = """
Using the exam paper content provided, answer the following question with precision:

{question}

Structure your answer as follows:
1. LIST each topic covered in the specified section, with the question number it appears in
2. MAP each topic to its official Assessment Objective (AO1/AO2/AO3/AO4 as relevant)
3. IDENTIFY the syllabus theme or period each topic falls under

Be specific — use the exact topic names from the AQA syllabus where possible.
"""

# Q2 type: Concept gap analysis from zero score
CONCEPT_GAP_PROMPT = """
A student scored 0 marks on the question described below. Using the mark scheme \
logic visible in the exam paper context, diagnose their likely knowledge gaps.

{question}

Structure your answer as follows:
1. MINIMUM THRESHOLD: What is the minimum a student must demonstrate to score any marks? \
   (Level 1 entry point per the mark scheme)
2. CONCEPT GAPS: List the specific historical knowledge or skill gaps a zero score indicates
3. MARK SCHEME SKILLS: Which Assessment Objective skills are entirely missing?
4. REVISION TARGETS: Name 2–3 specific topics or skills the student must address

Be direct — this student needs a clear diagnosis, not general encouragement.
"""

# Q3 type: Mark weighting and examiner priorities
MARK_WEIGHTING_PROMPT = """
Using the exam paper content provided, answer the following question:

{question}

Structure your answer as follows:
1. MARK TABLE: List all questions with their mark allocation (highest to lowest)
2. TOPIC PRIORITY: For each high-mark question, identify the syllabus topic it assesses
3. EXAMINER SIGNAL: What does this mark distribution tell us about what the examiner \
   considers most important in this topic area?
4. STUDENT IMPLICATION: Which topics should a student prioritise based on this weighting?
"""

# Q4 type: Practice question generation
PRACTICE_QUESTION_PROMPT = """
Using the exam paper content provided, generate practice questions as requested:

{question}

Requirements for each generated question:
1. SAME COMMAND WORDS as the original (e.g. 'Explain', 'Analyse', 'How does X differ')
2. SAME MARK ALLOCATION as the original question
3. SAME ASSESSMENT OBJECTIVE focus (AO4a, AO4b etc.)
4. SAME DIFFICULTY LEVEL — do not simplify
5. Include a brief MARK SCHEME OUTLINE for each generated question \
   showing Level 1 and Level 2 descriptors
6. Stay within the same syllabus period and topic area

Format each practice question clearly numbered (Practice Q1, Practice Q2, Practice Q3).
"""

# Q5 type: Revision priority order
REVISION_PRIORITY_PROMPT = """
Using the exam paper content provided, answer the following revision planning question:

{question}

Structure your answer as follows:
1. TOPIC INVENTORY: List all topics/skills tested in this paper with their mark weighting
2. PRIORITY ORDER: Rank topics from highest to lowest revision priority, \
   with a clear rationale for each ranking decision
3. 3-WEEK PLAN: Break the revision into Week 1, Week 2, Week 3 with specific focus areas
4. QUICK WINS: Identify any topics where a small amount of focused revision \
   yields disproportionate marks (e.g. high-mark topics with low conceptual complexity)
5. AO BALANCE: Ensure the plan covers all Assessment Objectives tested in this paper

Be practical — this student has limited time.
"""

# ─── Prompt selector ─────────────────────────────────────────────────────────

QUESTION_PROMPTS = {
    "syllabus_mapping":   SYLLABUS_MAPPING_PROMPT,
    "concept_gap":        CONCEPT_GAP_PROMPT,
    "mark_weighting":     MARK_WEIGHTING_PROMPT,
    "practice_questions": PRACTICE_QUESTION_PROMPT,
    "revision_priority":  REVISION_PRIORITY_PROMPT,
}


def classify_question(question_text: str) -> str:
    """
    Classify a query into one of the five prompt types.
    Keyword-based classification — deterministic and inspectable.
    """
    q = question_text.lower()

    if any(kw in q for kw in ["syllabus", "curriculum", "section", "topic", "cover", "map"]):
        return "syllabus_mapping"
    if any(kw in q for kw in ["scored 0", "zero", "concept gap", "gap", "failed", "what did"]):
        return "concept_gap"
    if any(kw in q for kw in ["mark weighting", "highest mark", "most marks", "priority", "examiner"]):
        # Distinguish mark weighting from revision priority
        if any(kw in q for kw in ["weeks", "revision", "revise", "plan", "prepare"]):
            return "revision_priority"
        return "mark_weighting"
    if any(kw in q for kw in ["practice question", "generate", "same style", "similar"]):
        return "practice_questions"
    if any(kw in q for kw in ["weeks", "revision", "revise", "plan", "prepare", "priorit"]):
        return "revision_priority"

    # Default to syllabus mapping for unclassified queries
    return "syllabus_mapping"


def build_prompt(question_text: str, context: str, board: str = "AQA",
                 qualification: str = "GCSE History") -> tuple[str, str]:
    """
    Build the full (system_prompt, user_prompt) pair for a query.
    Returns a tuple so callers can inspect both independently.
    """
    query_type = classify_question(question_text)
    system = SYSTEM_BASE.format(board=board, qualification=qualification, context=context)
    user_template = QUESTION_PROMPTS[query_type]
    user = user_template.format(question=question_text)

    return system, user, query_type
