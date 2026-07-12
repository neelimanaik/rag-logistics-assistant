from src.embeddings.embedder import embed_texts
from src.vectorstore.faiss_store import FaissStore
from src.retrieval.keyword_search import KeywordSearch


class Retriever:

    def __init__(self, index_path):
        self.store = FaissStore()
        self.store.load(index_path)
        self.keyword = KeywordSearch(f"{index_path}/metadata.json")

    def _rrf_fuse(self, result_lists, k_rrf=60):
        """Reciprocal Rank Fusion (RRF).

        Each source (vector search, keyword search) ranks its own hits. RRF
        combines them by RANK, not by raw score — so we never have to compare an
        L2 distance against a keyword overlap count (they are not comparable,
        which was the root of the old bug). A document's fused score is:

            sum over sources of  1 / (k_rrf + rank)

        Documents that rank well in BOTH sources rise to the top. k_rrf=60 is the
        standard constant; it softens the pull of any single #1 rank so one
        source can't completely dominate.
        """
        fused_score = {}
        best_copy = {}

        for results in result_lists:
            for rank, r in enumerate(results):
                key = r["text"]
                fused_score[key] = fused_score.get(key, 0.0) + 1.0 / (k_rrf + rank + 1)

                # Keep the copy that carries a real vector score, so the
                # confidence step downstream still has an L2 distance to use.
                existing = best_copy.get(key)
                if existing is None or (
                    existing.get("score") is None and r.get("score") is not None
                ):
                    best_copy[key] = r

        # Attach the fused score (handy for debugging / future ranking) and sort
        # best-first.
        for key, r in best_copy.items():
            r["rrf_score"] = fused_score[key]

        return sorted(best_copy.values(), key=lambda r: r["rrf_score"], reverse=True)

    def _dedup_by_section(self, results):
        """Drop near-duplicate passages that map to the same
        (document, section, page range), keeping the highest-ranked one.

        This removes the repeated citations we saw: vector and keyword search
        often return slightly different copies of the same passage, which dedup
        by exact text missed but this catches.
        """
        seen = set()
        unique = []
        for r in results:
            md = r["metadata"]
            identity = (
                md.get("source_file"),
                md.get("section"),
                md.get("page_start"),
                md.get("page_end"),
            )
            if identity not in seen:
                seen.add(identity)
                unique.append(r)
        return unique

    def query(self, question, k=10, filters=None, top_n=5):
        # 1) Ranked hits from each source.
        vector_results = self.store.search(embed_texts([question])[0], k)
        keyword_results = self.keyword.search(question, k)

        # 2) Fuse the two ranked lists with RRF (rank-based).
        fused = self._rrf_fuse([vector_results, keyword_results])

        # 3) Remove near-duplicate passages (same document/section/pages).
        fused = self._dedup_by_section(fused)

        # 4) Apply metadata filters from the router, if any. Fall back to the
        #    unfiltered set if filtering would leave nothing.
        if filters:
            filtered = [
                r for r in fused
                if all(r["metadata"].get(fk) == fv for fk, fv in filters.items())
            ]
            if filtered:
                return filtered[:top_n]

        return fused[:top_n]
