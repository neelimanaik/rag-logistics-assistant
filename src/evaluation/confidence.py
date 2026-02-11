def compute_confidence(results):

    if not results:
        return "LOW"

    # Average similarity score
    avg_score = sum(r["score"] for r in results) / len(results)

    # Section agreement
    sections = [r["metadata"].get("section") for r in results]
    agreement = len(set(sections)) <= 2

    # ----- Heuristic thresholds -----
    if avg_score < 0.5 and agreement:
        return "HIGH"

    if avg_score < 1.0:
        return "MEDIUM"

    return "LOW"
