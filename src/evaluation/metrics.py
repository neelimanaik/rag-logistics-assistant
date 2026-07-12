def precision_at_k(retrieved, ground_truth_sections):
    hits = 0
    for r in retrieved:
        if r["metadata"].get("section") in ground_truth_sections:
            hits += 1
    return hits / len(retrieved) if retrieved else 0