import faiss
import numpy as np
import json
import os


class FaissStore:
    """
    Lightweight FAISS wrapper storing:
    - vectors in FAISS index
    - metadata separately in JSON
    """

    def __init__(self, dim=None):
        self.index = None
        self.metadata = []
        self.dim = dim

        if dim is not None:
            self.index = faiss.IndexFlatL2(dim)

    # ---------- Add Vectors ----------
    def add(self, vectors, metadatas):
        np_vectors = np.array(vectors).astype("float32")

        if self.index is None:
            self.dim = np_vectors.shape[1]
            self.index = faiss.IndexFlatL2(self.dim)

        self.index.add(np_vectors)
        self.metadata.extend(metadatas)

    # ---------- Search ----------
    def search(self, query_vector, k=5):

        import numpy as np

        q = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(q, k)

        results = []

        for rank, i in enumerate(indices[0]):
            if i < len(self.metadata):
                results.append({
                    "metadata": self.metadata[i]["metadata"],
                    "text": self.metadata[i]["text"],
                    "score": float(distances[0][rank])
                })

        return results


    # ---------- Save ----------
    def save(self, path):
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, f"{path}/index.faiss")

        with open(f"{path}/metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    # ---------- Load ----------
    def load(self, path):
        self.index = faiss.read_index(f"{path}/index.faiss")

        with open(f"{path}/metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)
