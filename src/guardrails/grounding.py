import re

# A small stopword set so common filler words don't inflate the overlap score.
_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "which", "when",
    "your", "you", "are", "was", "were", "will", "would", "can", "could",
    "should", "have", "has", "had", "not", "but", "they", "them", "there",
    "then", "than", "into", "onto", "over", "under", "about", "these", "those",
    "such", "may", "might", "must", "shall", "been", "being", "also", "any",
    "all", "per", "our", "their", "its", "it", "is", "of", "to", "in", "on",
    "an", "as", "at", "or", "by", "be", "a",
}


def _content_words(text):
    """Distinct, meaningful words (lowercased, length > 3, non-stopword)."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if len(w) > 3 and w not in _STOPWORDS}


def check_grounding(answer, context, min_overlap=0.25):
    """Output guardrail: is the answer actually supported by the retrieved context?

    Heuristic: what fraction of the answer's distinctive words also appear in the
    context we gave the model? A grounded RAG answer reuses the source's
    vocabulary; an answer the model invented from its own memory will not.

    Returns (is_grounded, overlap_fraction).

    This is a cheap, deterministic first check — no extra LLM call — designed to
    catch obvious ungrounded/hallucinated answers. A stronger LLM-judge grounding
    check can be layered on later (B6, evaluation).
    """
    answer_words = _content_words(answer)
    if not answer_words:
        # Nothing substantive to verify (e.g. a very short or template reply).
        return True, 1.0

    context_words = _content_words(context)
    overlap = len(answer_words & context_words) / len(answer_words)
    return overlap >= min_overlap, overlap
