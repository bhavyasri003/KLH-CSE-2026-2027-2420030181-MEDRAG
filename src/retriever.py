import os
import re
import numpy as np
import pandas as pd
import faiss
import torch
from transformers import AutoTokenizer, AutoModel


# ==================================================
# PATHS
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "medquad_clean.csv"
)

INDEX_FILE = os.path.join(
    BASE_DIR,
    "vectorstore",
    "medquad.index"
)

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(DATA_FILE)

# Replace NaN values
df = df.fillna("")

print("MedQuAD records:", len(df))

# Precomputed lowercase question text, used by the lexical candidate
# search below. Built once at load time so repeated queries are cheap.
QUESTION_LOWER = df["question"].astype(str).str.lower()


# ==================================================
# LOAD BIOBERT
# ==================================================

print("Loading BioBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()

print("BioBERT loaded.")


# ==================================================
# LOAD FAISS
# ==================================================

index = faiss.read_index(INDEX_FILE)

print("FAISS vectors:", index.ntotal)


# ==================================================
# CREATE QUERY EMBEDDING
# (unchanged — not part of this fix's scope)
# ==================================================

def create_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(
        dim=1
    )

    embedding = embedding.numpy().astype(
        "float32"
    )

    faiss.normalize_L2(embedding)

    return embedding


# ==================================================
# QUERY INFORMATION
# ==================================================

def analyse_query(query):
    """
    Classify the user's medical question into a retrieval intent.

    The order is intentional:
    specific medical intents are checked before general
    definition/information questions.
    """

    query_lower = query.lower().strip()

    # Normalize punctuation/spacing
    query_lower = re.sub(r"[?!.,;:]+", " ", query_lower)
    query_lower = re.sub(r"\s+", " ", query_lower).strip()

    # --------------------------------------------------
    # SYMPTOMS
    # --------------------------------------------------
    if any(phrase in query_lower for phrase in [
        "symptom",
        "symptoms",
        "signs and symptoms",
        "signs of",
        "signs",
        "how does it feel",
        "what does it feel like",
        "indications",
        "warning signs",
        "clinical signs"
    ]):
        return "symptoms"

    # --------------------------------------------------
    # TREATMENT / MANAGEMENT
    # --------------------------------------------------
    if any(phrase in query_lower for phrase in [
        "treatment",
        "treatments",
        "treat",
        "treated",
        "cure",
        "cured",
        "manage",
        "managed",
        "management",
        "how can it be treated",
        "how is it treated",
        "how can it be managed",
        "what can be done"
    ]):
        return "treatment"

    # --------------------------------------------------
    # DIAGNOSIS / TESTING
    # --------------------------------------------------
    if any(phrase in query_lower for phrase in [
        "diagnosis",
        "diagnose",
        "diagnosed",
        "diagnostic",
        "test",
        "tests",
        "testing",
        "how is it diagnosed",
        "how do doctors diagnose",
        "what tests"
    ]):
        return "diagnosis"

    # --------------------------------------------------
    # GENETIC / INHERITANCE
    # --------------------------------------------------
    if any(phrase in query_lower for phrase in [
        "inherited",
        "inheritance",
        "genetic",
        "gene",
        "genes",
        "hereditary",
        "runs in families",
        "passed down"
    ]):
        return "genetic"

    # --------------------------------------------------
    # CAUSES
    # --------------------------------------------------
    if any(phrase in query_lower for phrase in [
        "what causes",
        "what cause",
        "causes of",
        "cause of",
        "caused by",
        "why does",
        "why do",
        "why is",
        "why are",
        "reason for",
        "reasons for",
        "what leads to",
        "what can cause"
    ]):
        return "causes"

    # --------------------------------------------------
    # PREVENTION
    # --------------------------------------------------
    if any(phrase in query_lower for phrase in [
        "prevent",
        "prevention",
        "avoid",
        "avoiding",
        "reduce the risk",
        "lower the risk",
        "how can i prevent",
        "how to prevent"
    ]):
        return "prevention"

    # --------------------------------------------------
    # COMPLICATIONS
    # --------------------------------------------------
    if any(phrase in query_lower for phrase in [
        "complication",
        "complications",
        "problems caused by",
        "what happens if",
        "what happens when",
        "long term effects",
        "long-term effects",
        "possible problems"
    ]):
        return "complications"

    # --------------------------------------------------
    # PROGNOSIS / OUTLOOK
    # --------------------------------------------------
    if any(phrase in query_lower for phrase in [
        "outlook",
        "prognosis",
        "life expectancy",
        "survival",
        "what is the outlook",
        "what happens over time"
    ]):
        return "prognosis"

    # --------------------------------------------------
    # DEFINITION / GENERAL INFORMATION
    # --------------------------------------------------
    if any(phrase in query_lower for phrase in [
        "what is",
        "what are",
        "what's",
        "tell me about",
        "define",
        "definition of",
        "explain",
        "what does",
        "what do you mean by"
    ]):
        return "definition"

    # --------------------------------------------------
    # FALLBACK
    # --------------------------------------------------
    return ""

# ==================================================
# GENERAL DEFINITION QUESTION DETECTOR
# (new — recognises MedQuAD's own "What is (are) X ?"
# style questions, as opposed to narrower subtype/
# complication phrasing like "X in Y" or "X caused by Y")
# ==================================================

DEFINITION_QUESTION_PATTERN = re.compile(
    r"^(what is \(are\)|what is|what are|what's)\s+(.+?)\s*\?\s*$"
)

DEFINITION_DISQUALIFIERS = [
    " of ",
    " in ",
    " for ",
    " related to ",
    " caused by ",
    " and ",
    ","
]


def is_general_definition_question(question):

    q = question.strip().lower()

    match = DEFINITION_QUESTION_PATTERN.match(q)

    if not match:
        return False

    topic_part = match.group(2)

    if any(d in topic_part for d in DEFINITION_DISQUALIFIERS):
        return False

    return True


# ==================================================
# EXTRACT IMPORTANT TERMS
# ==================================================

def get_medical_terms(query):

    words = re.findall(
        r"[a-zA-Z]+",
        query.lower()
    )

    stop_words = {
        "what",
        "are",
        "the",
        "is",
        "of",
        "for",
        "a",
        "an",
        "and",
        "to",
        "in",
        "on",
        "with",
        "how",
        "does",
        "do",
        "can",
        "tell",
        "me",
        "about",
        "please",
        "which",
        "what's"
    }

    terms = [
        word
        for word in words
        if word not in stop_words
        and len(word) > 2
    ]

    return terms

# ==================================================
# MEDICAL TOPIC EXTRACTION
# ==================================================

INTENT_PATTERNS = [
    r"^what are the symptoms of\s+",
    r"^what are the signs and symptoms of\s+",
    r"^what causes\s+",
    r"^what is the cause of\s+",
    r"^what are the causes of\s+",
    r"^what are the treatments for\s+",
    r"^what is the treatment for\s+",
    r"^how is\s+.*?\s+treated\s*$",
    r"^what is\s+",
    r"^what are\s+",
    r"^what's\s+",
    r"^tell me about\s+",
    r"^define\s+",
    r"^explain\s+"
]


def normalize_topic(text):
    """
    Normalize a medical topic while preserving multi-word
    entities such as 'common cold' and 'heart disease'.
    """
    text = str(text).lower().strip()

    # Remove MedQuAD-style punctuation
    text = re.sub(r"\(are\)", "", text)
    text = re.sub(r"[?!.,:;]+$", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_medical_topic(query):
    """
    Extract the medical topic from a natural-language question.

    Examples:
        What are the symptoms of hypertension?
        -> hypertension

        What causes migraine?
        -> migraine

        What is anemia?
        -> anemia

        What are the symptoms of common cold?
        -> common cold
    """

    topic = normalize_topic(query)

    # Remove the question mark first
    topic = topic.rstrip("?").strip()

    # Apply intent patterns
    for pattern in INTENT_PATTERNS:
        new_topic = re.sub(pattern, "", topic, count=1).strip()

        if new_topic != topic and new_topic:
            topic = new_topic
            break

    # Remove common trailing question words
    topic = re.sub(
        r"\b(please|for me|to know)\s*$",
        "",
        topic
    ).strip()

    return normalize_topic(topic)


def topic_match_score(query_topic, candidate_question):
    """
    Compare the extracted query topic against a dataset question.

    Returns:
        1.0 = exact topic match
        0.7 = candidate contains the topic as a larger phrase
        0.0 = no topic match
    """

    if not query_topic:
        return 0.0

    candidate = normalize_topic(candidate_question)

    # Extract candidate topic using the same intent removal logic
    candidate_topic = extract_medical_topic(candidate)

    if candidate_topic == query_topic:
        return 1.0

    # Multi-word topics such as 'common cold'
    # must appear as a complete phrase.
    if " " in query_topic:
        if re.search(
            r"\b" + re.escape(query_topic) + r"\b",
            candidate_topic
        ):
            return 0.7

    # Single-word topic: avoid giving exact credit to subtypes.
    # Example:
    # query = hypertension
    # candidate = pulmonary hypertension
    # This is only a partial/subtype match.
    if re.search(
        r"\b" + re.escape(query_topic) + r"\b",
        candidate_topic
    ):
        return 0.7

    return 0.0
# ==================================================
# WHOLE-WORD MATCH HELPER
# (new — prevents a term like "anemia" from matching
# inside an unrelated word like "hypertryptophanemia")
# ==================================================

def contains_whole_word(term, text):

    pattern = r"\b" + re.escape(term) + r"\b"

    return re.search(pattern, text) is not None


# ==================================================
# FULL-CORPUS LEXICAL CANDIDATE SEARCH
# (new — searches the QUESTION field only, across all
# 14,978 records, so a topic match isn't limited to
# whatever FAISS happened to put in its top-K)
# ==================================================

def lexical_candidates(query_terms):

    if not query_terms:
        return set()

    pattern = r"\b(?:" + "|".join(
        re.escape(term)
        for term in query_terms
    ) + r")\b"

    mask = QUESTION_LOWER.str.contains(
        pattern,
        regex=True,
        na=False
    )

    return set(df.index[mask].tolist())


# ==================================================
# SEARCH
# ==================================================

def search(query, top_k=5, faiss_candidate_k=1000):

    query_embedding = create_embedding(query)
    query_vec = query_embedding[0]

    # 1) Wider FAISS candidate pool (was 100, now up to 1000)
    candidate_k = min(faiss_candidate_k, len(df))

    scores, indices = index.search(
        query_embedding,
        candidate_k
    )

    faiss_score_map = {
        int(idx): float(score)
        for score, idx in zip(scores[0], indices[0])
        if idx >= 0
    }

    query_terms = get_medical_terms(query)
    question_type = analyse_query(query)
    query_topic = extract_medical_topic(query)

    # 2) Full-corpus lexical candidates over the QUESTION field
    lexical_idx = lexical_candidates(query_terms)

    # 3) Union of both candidate sources
    candidate_ids = set(faiss_score_map.keys()) | lexical_idx

    results = []

    for idx in candidate_ids:

        if idx < 0 or idx >= len(df):
            continue

        row = df.iloc[idx]

        question = str(row["question"])
        answer = str(row["answer"])

        question_lower = question.lower()
        answer_lower = answer.lower()

        # ------------------------------------------
        # Medical topic matching
        # ------------------------------------------

        topic_score = topic_match_score(
            query_topic,
            question
        )

        dataset_question_type = str(
            row.get("question_type", "")
        ).lower()

        # ------------------------------------------
        # Term matching — QUESTION field first.
        # A term only found in the (much longer) answer
        # counts far less, so a common word buried in a
        # long answer can't dominate ranking the way it
        # could before.
        # ------------------------------------------

        terms_in_question = [
            t for t in query_terms
            if contains_whole_word(t, question_lower)
        ]

        terms_in_answer_only = [
            t for t in query_terms
            if contains_whole_word(t, answer_lower)
            and t not in terms_in_question
        ]

        question_term_score = (
            len(terms_in_question) / max(len(query_terms), 1)
        )

        answer_term_score = (
            len(terms_in_answer_only) / max(len(query_terms), 1)
        )

        # ------------------------------------------
        # Question type matching (unchanged logic)
        # ------------------------------------------

        type_score = 0.0

        if question_type:

            if question_type == "definition":
                # MedQuAD's own question_type field uses "information"
                # for general definition-style records, not "definition".
                if dataset_question_type == "information":
                    type_score = 1.0
                elif question_type in question_lower:
                    type_score = 0.7

            elif question_type in dataset_question_type:
                type_score = 1.0

            elif question_type in question_lower:
                type_score = 0.7

        # ------------------------------------------
        # FAISS similarity — reused if this candidate came
        # from the FAISS search; otherwise computed directly
        # from the stored vector (candidate came from the
        # lexical search only, so it wasn't in the top-1000).
        # ------------------------------------------

        if idx in faiss_score_map:
            faiss_score = faiss_score_map[idx]
        else:
            stored_vec = index.reconstruct(int(idx))
            faiss_score = float(np.dot(stored_vec, query_vec))

        # ------------------------------------------
        # Exact medical-topic priority
        # ------------------------------------------

        exact_topic_bonus = (
            1.0
            if topic_score == 1.0
            else 0.0
        )

        subtype_penalty = (
            1.0
            if query_topic
            and topic_score == 0.7
            and " " not in query_topic
            else 0.0
        )

        # Small ranking preference for general "What is (are) X ?"
        # style records when the query itself is a definition-style
        # question. Only active for question_type == "definition",
        # so it has no effect on symptoms/causes/treatment/etc queries.
        definition_bonus = (
            1.0
            if question_type == "definition"
            and is_general_definition_question(question)
            else 0.0
        )

        # ------------------------------------------
        # Final hybrid score
        # ------------------------------------------

        combined_score = (
            0.40 * question_term_score
            + 0.20 * type_score
            + 0.20 * faiss_score
            + 0.08 * answer_term_score
            + 0.10 * exact_topic_bonus
            + 0.08 * definition_bonus
            - 0.06 * subtype_penalty
        )
        results.append({

            "score": combined_score,
            "topic": query_topic,
            "topic_score": topic_score,

            "faiss_score": faiss_score,

            "term_score": question_term_score,

            "answer_term_score": answer_term_score,

            "type_score": type_score,

            "question": question,

            "answer": answer,

            "source": str(
                row["document_source"]
            ),

            "url": str(
                row["document_url"]
            ),

            "category": str(
                row["category"]
            ),

            "question_type": dataset_question_type
        })

    # Sort by final score

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    test_queries = [

        "What are the symptoms of asthma?",

        "What causes diabetes?",

        "What are the symptoms of hypertension?"
    ]


    for query in test_queries:

        print("\n")

        print("=" * 70)

        print(
            "QUERY:",
            query
        )

        print("=" * 70)


        results = search(
            query,
            top_k=5
        )


        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nResult {i}"
            )

            print(
                "-" * 50
            )

            print(
                "Combined score:",
                round(
                    result["score"],
                    4
                )
            )

            print(
                "FAISS score:",
                round(
                    result["faiss_score"],
                    4
                )
            )

            print(
                "Question term score:",
                round(
                    result["term_score"],
                    4
                )
            )

            print(
                "Answer term score:",
                round(
                    result["answer_term_score"],
                    4
                )
            )

            print(
                "Type score:",
                round(
                    result["type_score"],
                    4
                )
            )

            print(
                "Question:",
                result["question"]
            )

            print(
                "Answer:",
                result["answer"][:400]
            )

            print(
                "Source:",
                result["source"]
            )

            print(
                "Question type:",
                result["question_type"]
            )