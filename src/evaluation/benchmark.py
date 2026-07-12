from src.rag.pipeline import RagAssistant
import os

if os.getenv("CI") == "true":
    print("skipping LLM Evaluation in CI environment")
    exit(0)

TEST_SET = [
    {
        "question": "How do I file FDA entry?",
        "expected_sections": ["Creating FDA Entry"]
    },
    {
        "question": "What regulation governs duty?",
        "expected_sections": ["Duty Assessment Rules"]
    }
]

def run_benchmark(index_path):
    assistant = RagAssistant(index_path)

    results = []

    for item in TEST_SET:
        _, citations, confidence = assistant.ask(item["question"])

        retrieved_sections = [c["section"] for c in citations]

        score = precision_at_k(
            [{"metadata": {"section": s}} for s in retrieved_sections],
            item["expected_sections"]
        )

        results.append(score)

    print("Average Precision:", sum(results) / len(results))