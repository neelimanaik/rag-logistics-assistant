from src.llm.client import generate_answer
from src.llm.router_prompt import ROUTER_PROMPT


def classify_query(question):

    response = generate_answer(
        ROUTER_PROMPT,
        question
    )

    label = response.strip().lower()

    if "functional" in label:
        return {"document_type": "user_manual"}

    if "regulatory" in label:
        return {"document_type": "customs_regulation"}

    return None
