import json
import re


class KeywordSearch:

    def __init__(self, metadata_path):
        with open(metadata_path, encoding="utf-8") as f:
            self.docs = json.load(f)

    def tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def search(self, query, k=5):

        q_tokens = set(self.tokenize(query))
        scores = []

        for doc in self.docs:
            tokens = set(self.tokenize(doc["text"]))
            overlap = len(q_tokens.intersection(tokens))

            if overlap > 0:
                scores.append((overlap, doc))

        scores.sort(reverse=True, key=lambda x: x[0])

        # Return results in the SAME shape as the vector store
        # ({"metadata", "text", "score"}) so downstream code can treat
        # vector and keyword hits identically.
        #
        # score is set to None on purpose: a keyword "overlap count" is not
        # comparable to the vector store's L2 distance, so we don't pretend it
        # is. We keep the raw overlap under "keyword_overlap" for later use
        # (true score fusion is a Stage-B upgrade). None means "no comparable
        # similarity score", and confidence scoring is taught to skip it.
        return [
            {
                "metadata": doc["metadata"],
                "text": doc["text"],
                "score": None,
                "keyword_overlap": overlap,
            }
            for overlap, doc in scores[:k]
        ]
