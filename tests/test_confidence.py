"""Unit tests for the confidence heuristic (no model/index needed)."""

from src.evaluation.confidence import compute_confidence


def test_empty_results_is_low():
    assert compute_confidence([]) == "LOW"


def test_keyword_only_results_is_low():
    # Keyword hits carry score=None -> no comparable distance -> LOW.
    results = [{"metadata": {"section": "A"}, "text": "x", "score": None}]
    assert compute_confidence(results) == "LOW"


def test_close_match_is_high():
    # Best (smallest) distance is 0.47, under HIGH threshold 0.50.
    results = [
        {"metadata": {"section": "A"}, "text": "x", "score": 0.47},
        {"metadata": {"section": "B"}, "text": "y", "score": 0.90},
    ]
    assert compute_confidence(results) == "HIGH"


def test_medium_match():
    results = [{"metadata": {"section": "A"}, "text": "x", "score": 0.70}]
    assert compute_confidence(results) == "MEDIUM"


def test_far_match_is_low():
    results = [{"metadata": {"section": "A"}, "text": "x", "score": 1.20}]
    assert compute_confidence(results) == "LOW"
