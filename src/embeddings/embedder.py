import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------- Azure OpenAI Client ----------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview"
)

EMBED_MODEL = os.getenv("AZURE_EMBED_MODEL")


# ---------- Embedding Function ----------
def embed_texts(texts, batch_size=20):
    """
    Generate embeddings for a list of texts.
    Handles batching to avoid API limits.
    """

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )

        batch_embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
