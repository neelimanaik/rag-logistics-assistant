"""Retrieval-quality metrics.

Section names in the corpus are messy ("FAQ", "CODE PER PGA", "Page 14"), so we
match an expected label against a retrieved section by case-insensitive
SUBSTRING — expected labels act as keywords rather than exact strings.
"""


def _matches(section, expected_labels):
    """True if any expected label appears (case-insensitively) in `section`."""
    s = (section or "").lower()
    return any(label.lower() in s for label in expected_labels)


def precision_at_k(retrieved_sections, expected_labels):
    """Of the sections we retrieved, what fraction are relevant?

    High precision = we didn't return much junk.
    """
    if not retrieved_sections:
        return 0.0
    hits = sum(1 for s in retrieved_sections if _matches(s, expected_labels))
    return hits / len(retrieved_sections)


def recall_at_k(retrieved_sections, expected_labels):
    """Of the sections we expected, what fraction did we actually retrieve?

    High recall = we didn't miss the relevant material.
    """
    if not expected_labels:
        return 0.0
    found = sum(
        1
        for label in expected_labels
        if any(label.lower() in (s or "").lower() for s in retrieved_sections)
    )
    return found / len(expected_labels)
