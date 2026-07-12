"""Retrieval-quality benchmark.

Runs a small labelled set of questions through the retriever and reports
Precision@K and Recall@K per question plus the averages. We evaluate RETRIEVAL
(not generation) on purpose: retrieval quality is what these metrics measure, it
needs no LLM call, and it's fast and repeatable — so you can re-run it after every
retrieval change (e.g. the RRF work) and see whether quality moved.

Run it locally (needs the built index + Ollama for query embeddings):

    python -m src.evaluation.benchmark

The `expected_labels` are keyword hints matched as case-insensitive substrings of
retrieved section names. This is a STARTER set — expand it as you learn which
sections truly answer each question.
"""
from src.config.settings import settings
from src.retrieval.retriever import Retriever
from src.evaluation.metrics import precision_at_k, recall_at_k

EVAL_SET = [
    {
        "question": "When should APHIS Core be disclaimed?",
        "expected_labels": ["FAQ", "CODE PER PGA", "Disclaim"],
    },
    {
        "question": "What is an HTS duty exemption?",
        "expected_labels": ["HTS", "Duty Free Entry Certificate", "Exemption"],
    },
    {
        "question": "What are the APHIS Core disclaim codes?",
        "expected_labels": ["CODE PER PGA", "Disclaim"],
    },
    {
        "question": "How do I file an FDA entry?",
        "expected_labels": ["FDA", "Entry"],
    },
]


def run_benchmark(index_path=None, k=5):
    index_path = index_path or settings.index_path
    retriever = Retriever(index_path)

    precisions = []
    recalls = []

    print(f"\n{'Question':<48} P@{k}   R@{k}")
    print("-" * 66)

    for item in EVAL_SET:
        results = retriever.query(item["question"], k=10, top_n=k)
        retrieved_sections = [r["metadata"].get("section") for r in results]

        p = precision_at_k(retrieved_sections, item["expected_labels"])
        r = recall_at_k(retrieved_sections, item["expected_labels"])
        precisions.append(p)
        recalls.append(r)

        print(f"{item['question'][:46]:<48} {p:.2f}  {r:.2f}")

    print("-" * 66)
    mean_p = sum(precisions) / len(precisions)
    mean_r = sum(recalls) / len(recalls)
    print(f"{'MEAN':<48} {mean_p:.2f}  {mean_r:.2f}\n")
    return mean_p, mean_r


if __name__ == "__main__":
    settings.validate()
    run_benchmark()
