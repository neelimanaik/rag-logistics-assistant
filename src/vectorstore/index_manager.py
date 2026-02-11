import json
from src.embeddings.embedder import embed_texts
from src.vectorstore.faiss_store import FaissStore


def build_index(chunk_file, save_path):

    with open(chunk_file, encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    print("🔹 Generating embeddings...")
    vectors = embed_texts(texts)

    dim = len(vectors[0])
    store = FaissStore(dim)

    print("🔹 Building FAISS index...")
    store.add(vectors, chunks)

    store.save(save_path)

    print("✅ Index built and saved")
