""" 
EVALUATION

We decided to implement the following evaluation framework:
- Retrieval metrics (Precision@K, Recall@K and MRR)
- Generation metrics (faithfulness, relevance, correctness)
- Retrieval and generation latency
- Source evaluation
"""

# CONNECTION TO RAG PIPELINE AND EVALUATION QUESTIONS

import time
import csv
from statistics import mean

try:
    from rag_pipeline import retrieve, answer_question
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


EVAL_SET = [
    {
        "question": "What is DRS in Formula 1?",
        "expected_sources": ["f1_rules.txt"],
        "ground_truth_answer": "DRS is a system that reduces drag by opening a flap on the rear wing in designated zones."
    },
    {
        "question": "How does Formula 1 qualifying work?",
        "expected_sources": ["f1_rules.txt"],
        "ground_truth_answer": "Formula 1 qualifying determines the starting grid and is typically divided into Q1, Q2, and Q3."
    },
    {
        "question": "What is an undercut strategy?",
        "expected_sources": ["f1_strategy.md"],
        "ground_truth_answer": "An undercut is when a driver pits earlier to use fresher tyres and gain time over a rival."
    },
    {
        "question": "What is an overcut strategy?",
        "expected_sources": ["f1_strategy.md"],
        "ground_truth_answer": "An overcut is when a driver stays out longer than a rival to gain time before pitting."
    },
    {
        "question": "What happens during a safety car period?",
        "expected_sources": ["f1_rules.txt", "f1_strategy.md"],
        "ground_truth_answer": "During a safety car period, cars slow down and follow the safety car while track hazards are managed."
    },
    {
        "question": "What is the constructor championship?",
        "expected_sources": ["f1_history.txt", "f1_rules.txt"],
        "ground_truth_answer": "The constructor championship ranks teams based on points scored by their drivers."
    },
    {
        "question": "Who has the most wins in the driver statistics dataset?",
        "expected_sources": ["driver_stats.csv"],
        "ground_truth_answer": "The answer depends on the driver statistics dataset used."
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
"""
    Produces a CSV-ready table for manual qualitative evaluation.

    Fill the following fields manually:
    - faithfulness_score_1_to_5
    - relevance_score_1_to_5
    - correctness_score_1_to_5

    Suggested scale:
    1 = poor
    3 = good
    5 = strong
    """
def evaluate_generation_metrics(eval_set=EVAL_SET, top_k=5):
    results = []

    for item in eval_set:
        question = item["question"]
        ground_truth = item["ground_truth_answer"]

        response = answer_question(question, top_k=top_k)

        results.append({
            "question": question,
            "ground_truth_answer": ground_truth,
            "generated_answer": response.get("answer", ""),
            "sources": response.get("sources", []),
            "faithfulness_score_1_to_5": "",
            "relevance_score_1_to_5": "",
            "correctness_score_1_to_5": "",
            "qualitative_notes": ""
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

        contains_expected_source = any(
            source in returned_sources
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


