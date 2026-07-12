import uuid

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks for RAG.
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks


def enrich_chunks(chunks, doc_id):
    """
    Adds metadata to each chunk.
    """
    enriched = []

    for idx, chunk in enumerate(chunks):
        enriched.append({
            "chunk_id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "chunk_index": idx,
            "text": chunk,
            "metadata": {
                "source_type": "customs_regulation",
                "authority": "CBP",
                "country": "US",
                "application": "eBrokerage",
                "confidence_level": "high",
                "is_active": True
            }
        })

    return enriched
