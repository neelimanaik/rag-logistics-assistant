"""One-command index build:  PDFs -> chunks -> embeddings -> FAISS index.

Run this once before starting the API:

    python ingest.py
"""
from src.config.settings import settings
from main import run_ingestion_pipeline
from src.vectorstore.index_manager import build_index

if __name__ == "__main__":
    settings.validate()
    print(f"Provider={settings.llm_provider}  chat={settings.chat_model}  embed={settings.embed_model}")
    run_ingestion_pipeline()
    build_index("data/processed/chunks/all_chunks.json", settings.index_path)
    print(f"Index built at: {settings.index_path}")
