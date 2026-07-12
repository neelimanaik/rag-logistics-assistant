# Confidence thresholds are L2 distances (smaller = closer match). They are
# calibrated for the local `nomic-embed-text` model, where strong matches land
# around 0.45-0.6 (confirmed via debug_scores.py). They are model-specific and
# will move into config / the evaluation framework (B6) rather than living as
# constants here.
HIGH_MAX_DISTANCE = 0.50    # at least one very close match exists
MEDIUM_MAX_DISTANCE = 0.80  # at least one reasonably close match exists


def compute_confidence(results):
    """Rate retrieval confidence as HIGH / MEDIUM / LOW.

    We judge on the BEST (smallest) vector distance among the results: if the
    closest retrieved passage sits very near the query, the answer is likely in
    the context we are about to send the LLM. Keyword-only hits carry
    score=None and are skipped, because an overlap count is not a distance.

    Why this replaces the old heuristic: the previous version averaged distances
    AND required the results to come from <= 2 sections. That section rule made
    HIGH almost unreachable, because a good retrieval legitimately spans several
    sections. Judging on the best match is simpler, interpretable, and lets a
    genuinely strong match earn HIGH.
    """
    if not results:
        return "LOW"

    distances = [
        r["score"] for r in results if isinstance(r.get("score"), (int, float))
    ]
    if not distances:
        # Only keyword hits -> no comparable distance -> be cautious.
        return "LOW"

    best = min(distances)  # closest match = smallest L2 distance

    if best < HIGH_MAX_DISTANCE:
        return "HIGH"
    if best < MEDIUM_MAX_DISTANCE:
        return "MEDIUM"
    return "LOW"
