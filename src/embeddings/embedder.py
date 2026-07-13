from src.config.settings import settings
from src.llm.provider import get_client

# Same provider seam as the chat client, so chat and embeddings always use the
# same backend and configuration.
client = get_client()
EMBED_MODEL = settings.embed_model


def embed_texts(texts, batch_size=20):
    """Generate embeddings for a list of texts, batched to avoid API limits."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )
        batch_embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings
