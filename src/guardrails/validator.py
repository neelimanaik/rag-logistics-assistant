import re

# --- Prompt-injection / jailbreak patterns ---
# These target attempts to make the model ignore its instructions or leak its
# system prompt. Patterns are matched against the lower-cased question. They are
# intentionally specific to avoid blocking legitimate logistics phrasing (e.g.
# "act as importer of record" must NOT trip a rule).
INJECTION_PATTERNS = [
    r"ignore .*(previous|prior|above|earlier).*instruction",
    r"disregard .*(previous|prior|above|earlier).*(instruction|rule)",
    r"forget .*(previous|above|your).*(instruction|rule)",
    r"reveal .*(system )?(prompt|instruction)",
    r"show .*(your|the).*(system )?(prompt|instruction)",
    r"system prompt",
    r"you are now ",
    r"pretend (to be|you are)",
    r"developer mode",
    r"do anything now",
    r"new instructions?:",
    r"override .*(instruction|rule|guardrail)",
]

# --- Out-of-scope topics (kept conservative so real customs topics like
# "weapons imports" are not falsely blocked) ---
OUT_OF_SCOPE_TOPICS = ["politics", "religion", "violence"]

OUT_OF_SCOPE_MESSAGE = "This assistant only supports logistics and customs queries."
INJECTION_MESSAGE = (
    "Request blocked: it looks like an attempt to change the assistant's instructions."
)


def _matches_any(text, patterns):
    return any(re.search(pattern, text) for pattern in patterns)


def validate_query(question):
    """Input guardrail. Returns (is_allowed, reason_message).

    Runs BEFORE any retrieval or LLM call (see pipeline._prepare), so a bad
    request costs nothing. Two checks, in order:
      1. Prompt-injection / jailbreak attempts (pattern based).
      2. Out-of-scope topics.
    Returns (True, None) when the query is acceptable.
    """
    if not question or not question.strip():
        return False, OUT_OF_SCOPE_MESSAGE

    text = question.lower()

    if _matches_any(text, INJECTION_PATTERNS):
        return False, INJECTION_MESSAGE

    if any(topic in text for topic in OUT_OF_SCOPE_TOPICS):
        return False, OUT_OF_SCOPE_MESSAGE

    return True, None
