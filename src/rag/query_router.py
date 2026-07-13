from src.llm.client import generate_answer
from src.llm.router_prompt import ROUTER_PROMPT

# Cheap keyword hints for rule-based routing (no LLM call).
_FUNCTIONAL_HINTS = [
    "how do i",
    "how to",
    "steps",
    "procedure",
    "screen",
    "field",
    "tab",
    "click",
    "button",
    "navigate",
    "user manual",
    "enter the",
    "fill in",
]
_REGULATORY_HINTS = [
    "regulation",
    "law",
    "duty",
    "tariff",
    "hts",
    "compliance",
    "cfr",
    "fda",
    "aphis",
    "pga",
    "customs",
    "cbp",
    "admissib",
    "disclaim",
    "exemption",
    "prohibited",
    "restricted",
    "certificate",
]


def _rule_based_label(question):
    """Classify with keyword hints. Returns 'functional', 'regulatory', or None
    (inconclusive)."""
    text = question.lower()
    functional_hits = sum(hint in text for hint in _FUNCTIONAL_HINTS)
    regulatory_hits = sum(hint in text for hint in _REGULATORY_HINTS)

    if functional_hits == 0 and regulatory_hits == 0:
        return None
    return "regulatory" if regulatory_hits >= functional_hits else "functional"


def _label_to_filter(label):
    if label == "functional":
        return {"document_type": "user_manual"}
    if label == "regulatory":
        return {"document_type": "customs_regulation"}
    return None


def classify_query(question, use_llm_fallback=False):
    """Route a query to a document-type filter.

    The original version made a full LLM call (~10s in local testing, per our
    telemetry) just to pick a metadata filter — the biggest non-generation cost
    in the request. We now classify with cheap keyword rules first, which is
    effectively free and, for this domain, just as accurate.

    An LLM fallback is available for genuinely ambiguous queries but is OFF by
    default, so the hot path never pays for it. The filter is a soft hint anyway:
    the retriever falls back to unfiltered results if the filter matches nothing.
    """
    label = _rule_based_label(question)

    if label is None and use_llm_fallback:
        response = generate_answer(ROUTER_PROMPT, question)
        text = response.strip().lower()
        if "functional" in text:
            label = "functional"
        elif "regulatory" in text:
            label = "regulatory"

    return _label_to_filter(label)
