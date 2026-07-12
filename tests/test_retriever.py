"""Unit tests for RRF fusion and section dedup (the B2 upgrade).

We bypass Retriever.__init__ so no FAISS index on disk is required; we only
exercise the pure fusion/dedup logic.
"""
from src.retrieval.retriever import Retriever


def _bare_retriever():
    return Retriever.__new__(Retriever)


def test_rrf_prefers_docs_ranked_well_in_both_sources():
    r = _bare_retriever()
    vector = [
        {"text": "A", "metadata": {}, "score": 0.4},
        {"text": "B", "metadata": {}, "score": 0.5},
    ]
    keyword = [
        {"text": "B", "metadata": {}, "score": None},
        {"text": "C", "metadata": {}, "score": None},
    ]
    fused = r._rrf_fuse([vector, keyword])
    # B is present in BOTH lists, so it should rank first.
    assert fused[0]["text"] == "B"
    assert all("rrf_score" in item for item in fused)


def test_rrf_keeps_the_copy_with_a_real_score():
    r = _bare_retriever()
    vector = [{"text": "B", "metadata": {}, "score": 0.5}]
    keyword = [{"text": "B", "metadata": {}, "score": None}]
    fused = r._rrf_fuse([vector, keyword])
    assert len(fused) == 1
    # Kept the vector copy (real score) rather than the keyword copy (None).
    assert fused[0]["score"] == 0.5


def test_dedup_by_section_removes_duplicate_passages():
    r = _bare_retriever()
    md = {"source_file": "doc.pdf", "section": "FAQ", "page_start": 15, "page_end": 15}
    other = {"source_file": "doc.pdf", "section": "Intro", "page_start": 1, "page_end": 2}
    results = [
        {"text": "one copy", "metadata": md, "score": 0.4},
        {"text": "near dup", "metadata": md, "score": 0.5},   # same doc/section/pages
        {"text": "different", "metadata": other, "score": 0.6},
    ]
    deduped = r._dedup_by_section(results)
    assert len(deduped) == 2
    assert deduped[0]["text"] == "one copy"  # first occurrence kept
