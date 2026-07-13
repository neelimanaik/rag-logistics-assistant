"""Unit tests for keyword search result shape (the bug-2 fix)."""

import json

from src.retrieval.keyword_search import KeywordSearch


def _make_store(tmp_path):
    docs = [
        {"text": "FDA entry filing procedure", "metadata": {"section": "FDA"}},
        {"text": "HTS duty exemption rules", "metadata": {"section": "HTS"}},
    ]
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(docs), encoding="utf-8")
    return KeywordSearch(str(path))


def test_results_have_uniform_shape(tmp_path):
    ks = _make_store(tmp_path)
    results = ks.search("HTS duty", k=5)
    assert results  # found at least one match
    for r in results:
        # Same keys the vector store returns, plus keyword_overlap.
        assert {"metadata", "text", "score", "keyword_overlap"}.issubset(r.keys())
        # Keyword hits carry no comparable distance.
        assert r["score"] is None


def test_no_match_returns_empty(tmp_path):
    ks = _make_store(tmp_path)
    assert ks.search("zzznomatch", k=5) == []
