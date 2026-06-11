""" 
EVALUATION

We decided to implement the following evaluation framework:
- Retrieval metrics (Precision@K, Recall@K and MRR)
- Generation metrics (faithfulness, relevance, correctness)
- Retrieval and generation latency
- Source evaluation
"""

# CONNECTION TO RAG PIPELINE AND EVALUATION QUESTIONS

import re
import time
from statistics import mean

import numpy as np

try:
    from rag_pipeline import retrieve, answer_question, _get_pipeline
except ImportError:
    def retrieve(query: str, top_k: int = 5):
        print("WARNING: rag_pipeline.retrieve() is not connected yet.")
        return []

    def answer_question(query: str, top_k: int = 5):
        print("WARNING: rag_pipeline.answer_question() is not connected yet.")
        return {
            "answer": "RAG pipeline not connected yet.",
            "sources": [],
            "contexts": []
        }

"""
The evaluation dataset was designed to cover multiple Formula One knowledge domains.
"""
EVAL_SET = [
    {
        "question": "How many Formula One World Drivers' Championship titles has Max Verstappen won, and during which years?",
        "expected_sources": ["max_verstappen_chunks.json"],
        "ground_truth_answer": (
            "Max Verstappen has won four Formula One World Drivers' Championship "
            "titles consecutively from 2021 to 2024."
        ),
        "evaluation_focus": "Driver achievements and factual accuracy"
    },

    {
        "question": "What is the Circuit de Monaco and what major racing events are held there?",
        "expected_sources": ["circuit_de_monaco_chunks.json"],
        "ground_truth_answer": (
            "The Circuit de Monaco is a 3.337 km street circuit located on the "
            "streets of Monte Carlo and La Condamine in Monaco. It hosts the "
            "Formula One Monaco Grand Prix, Formula E Monaco ePrix, and the "
            "Historic Grand Prix of Monaco."
        ),
        "evaluation_focus": "Circuit characteristics and event identification"
    },

    {
        "question": "What type of engines are currently used in Formula One?",
        "expected_sources": ["formula_one_engines_chunks.json"],
        "ground_truth_answer": (
            "Formula One currently uses 1.6-litre four-stroke turbocharged "
            "90-degree V6 double-overhead camshaft hybrid power units, "
            "introduced in 2014."
        ),
        "evaluation_focus": "Technical specifications retrieval"
    },

    {
        "question": "When was the modern Formula One World Championship established?",
        "expected_sources": ["history_of_formula_one_chunks.json"],
        "ground_truth_answer": (
            "The foundation of modern Formula One began with FIA standardisation "
            "of rules in 1946, followed by the first World Championship of Drivers "
            "in 1950."
        ),
        "evaluation_focus": "Historical information retrieval"
    },

    {
        "question": "What are the primary goals of Formula One regulations?",
        "expected_sources": ["formula_one_regulations_chunks.json"],
        "ground_truth_answer": (
            "The primary goals of Formula One regulations are to ensure driver "
            "safety and fairness in competition, while also promoting "
            "environmental sustainability."
        ),
        "evaluation_focus": "Regulations understanding and synthesis"
    }
]


# RETRIEVAL METRICS

"""
Extract source names from retrieved chunks.
"""
def extract_sources(retrieved_chunks):
    sources = []

    for chunk in retrieved_chunks:
        source = chunk.get("source")
        if source:
            sources.append(source)

    return sources


"""
Precision@K:
Of the top-K retrieved sources, how many are relevant?
"""
def precision_at_k(retrieved_sources, expected_sources, k):
    retrieved_k = retrieved_sources[:k]

    if not retrieved_k:
        return 0.0

    relevant_count = sum(
        1 for source in retrieved_k
        if source in expected_sources
    )

    return relevant_count / len(retrieved_k)


"""
Recall@K:
Of the expected relevant sources, how many were retrieved in top-K?
"""
def recall_at_k(retrieved_sources, expected_sources, k):
    retrieved_k = retrieved_sources[:k]

    if not expected_sources:
        return 0.0

    matched_count = sum(
        1 for source in expected_sources
        if source in retrieved_k
    )

    return matched_count / len(expected_sources)


"""
MRR component:
Returns 1/rank of the first relevant retrieved source.
"""
def reciprocal_rank(retrieved_sources, expected_sources):
    for index, source in enumerate(retrieved_sources):
        if source in expected_sources:
            return 1 / (index + 1)

    return 0.0


"""
Runs Precision@K, Recall@K, and MRR over the evaluation set.
"""
def evaluate_retrieval_metrics(eval_set=EVAL_SET, top_k=5):
    results = []

    for item in eval_set:
        question = item["question"]
        expected_sources = item["expected_sources"]

        retrieved_chunks = retrieve(question, top_k=top_k)
        retrieved_sources = extract_sources(retrieved_chunks)

        results.append({
            "question": question,
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
            f"precision@{top_k}": precision_at_k(
                retrieved_sources,
                expected_sources,
                top_k
            ),
            f"recall@{top_k}": recall_at_k(
                retrieved_sources,
                expected_sources,
                top_k
            ),
            "mrr": reciprocal_rank(
                retrieved_sources,
                expected_sources
            )
        })

    return results


# GENERATION METRICS

def _scale_to_1_5(value: float) -> int:
    """Scale a 0-1 float to a 1-5 integer score."""
    return round(1 + max(0.0, min(1.0, value)) * 4)


def _faithfulness_score(answer: str, contexts: list[str]) -> int:
    """
    Faithfulness: fraction of answer tokens that appear in the retrieved contexts.
    Measures whether the answer is grounded in the context (not hallucinated).
    """
    if not answer or not contexts:
        return 1
    answer_tokens = set(re.sub(r"[^\w]", " ", answer.lower()).split())
    context_tokens = set(re.sub(r"[^\w]", " ", " ".join(contexts).lower()).split())
    if not answer_tokens:
        return 1
    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    return _scale_to_1_5(overlap)


def _relevance_score(question: str, answer: str) -> int:
    """
    Relevance: cosine similarity between question and answer embeddings.
    Measures whether the answer actually addresses the question.
    """
    if not answer:
        return 1
    try:
        embedder = _get_pipeline()._store._embedder
        vecs = embedder.encode([question, answer], convert_to_numpy=True)
        q_vec = vecs[0] / (np.linalg.norm(vecs[0]) + 1e-9)
        a_vec = vecs[1] / (np.linalg.norm(vecs[1]) + 1e-9)
        similarity = float(np.dot(q_vec, a_vec))
        return _scale_to_1_5(similarity)
    except Exception:
        return 1


def _correctness_score(generated: str, ground_truth: str) -> int:
    """
    Correctness: cosine similarity between generated answer and ground truth.
    Measures how close the answer is to the expected answer.
    """
    if not generated or not ground_truth:
        return 1
    try:
        embedder = _get_pipeline()._store._embedder
        vecs = embedder.encode([generated, ground_truth], convert_to_numpy=True)
        g_vec = vecs[0] / (np.linalg.norm(vecs[0]) + 1e-9)
        t_vec = vecs[1] / (np.linalg.norm(vecs[1]) + 1e-9)
        similarity = float(np.dot(g_vec, t_vec))
        return _scale_to_1_5(similarity)
    except Exception:
        return 1


def evaluate_generation_metrics(eval_set=EVAL_SET, top_k=5):
    """
    Runs generation metrics over the evaluation set.

    Scores (1–5 scale, 5 = best):
    - faithfulness:  answer grounded in retrieved context (token overlap)
    - relevance:     answer addresses the question (embedding similarity)
    - correctness:   answer matches ground truth (embedding similarity)
    """
    results = []

    for item in eval_set:
        question    = item["question"]
        ground_truth = item["ground_truth_answer"]

        response = answer_question(question, top_k=top_k)
        answer   = response.get("answer", "")
        contexts = response.get("contexts", [])

        results.append({
            "question":                  question,
            "ground_truth_answer":       ground_truth,
            "generated_answer":          answer,
            "sources":                   response.get("sources", []),
            "faithfulness_score_1_to_5": _faithfulness_score(answer, contexts),
            "relevance_score_1_to_5":    _relevance_score(question, answer),
            "correctness_score_1_to_5":  _correctness_score(answer, ground_truth),
        })

    return results


# RETRIEVAL AND GENERATION LATENCY
"""
    Measures retrieval time and generation time separately.
    """
def evaluate_latency(eval_set=EVAL_SET, top_k=5):
    results = []

    for item in eval_set:
        question = item["question"]

        retrieval_start = time.time()
        retrieve(question, top_k=top_k)
        retrieval_latency = time.time() - retrieval_start

        generation_start = time.time()
        answer_question(question, top_k=top_k)
        generation_latency = time.time() - generation_start

        results.append({
            "question": question,
            "retrieval_latency_seconds": retrieval_latency,
            "generation_latency_seconds": generation_latency
        })

    return results


# SOURCE EVALUATION
"""
    Checks whether generated answers include sources and whether
    at least one expected source appears in the answer sources.
    """
def evaluate_sources_and_citations(eval_set=EVAL_SET, top_k=5):
    results = []

    for item in eval_set:
        question = item["question"]
        expected_sources = item["expected_sources"]

        response = answer_question(question, top_k=top_k)
        returned_sources = response.get("sources", [])

        has_sources = len(returned_sources) > 0

        # returned_sources is a list of dicts with a "title" field
        # derive filename from title to match expected_sources format
        import re
        returned_filenames = [
            re.sub(r"[^\w\-]", "_", s["title"]).lower()[:100] + "_chunks.json"
            for s in returned_sources
            if isinstance(s, dict) and "title" in s
        ]

        contains_expected_source = any(
            source in returned_filenames
            for source in expected_sources
        )

        results.append({
            "question": question,
            "expected_sources": expected_sources,
            "returned_sources": returned_sources,
            "has_sources": has_sources,
            "contains_expected_source": contains_expected_source
        })

    return results


