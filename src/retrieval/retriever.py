from src.embeddings.embedder import embed_texts
from src.vectorstore.faiss_store import FaissStore
from src.retrieval.keyword_search import KeywordSearch



class Retriever:

    def __init__(self, index_path):

        self.store = FaissStore()
        self.store.load(index_path)

        self.keyword = KeywordSearch(
            f"{index_path}/metadata.json"
        )


    def query(self, question, k=10, filters=None):

        # Vector search
        vector_results = self.store.search(
            embed_texts([question])[0],
            k
        )

        # Keyword search
        keyword_results = self.keyword.search(question, k)

        # Merge results
        combined = vector_results + keyword_results

        # Deduplicate by text
        seen = set()
        unique = []

        for r in combined:
            text = r["text"]
            if text not in seen:
                seen.add(text)
                unique.append(r)

        # Apply metadata filters
        if filters:
            filtered = [
                r for r in unique
                if all(r["metadata"].get(k) == v for k, v in filters.items())
            ]
            if filtered:
                return filtered[:5]

        return unique[:5]
