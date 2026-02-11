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

        return [d[1] for d in scores[:k]]
