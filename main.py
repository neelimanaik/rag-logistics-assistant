from dotenv import load_dotenv
import os

#from evaluation import confidence
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))


print("KEY LOADED:", os.getenv("AZURE_OPENAI_KEY") is not None)
import json

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.document_router import infer_document_metadata
from src.preprocessing.structure_chunker import structure_aware_chunk
#for indexing
from src.vectorstore.index_manager import build_index

from src.rag.pipeline import RagAssistant




RAW_DATA_ROOT = "data/raw"
OUTPUT_PATH = "data/processed/chunks"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def run_ingestion_pipeline():
    all_chunks = []

    for root, _, files in os.walk(RAW_DATA_ROOT):
        for file in files:
            if not file.lower().endswith(".pdf"):
                continue

            file_path = os.path.join(root, file)
            print(f"📄 Processing: {file_path}")

            base_metadata = infer_document_metadata(file_path)
            pages = load_pdf(file_path)

            chunks = structure_aware_chunk(pages)

            for c in chunks:
                c.metadata.update(base_metadata)
                c.metadata["source_file"] = file
                all_chunks.append({
                    "text": c.page_content,
                    "metadata": c.metadata
                })

    output_file = os.path.join(OUTPUT_PATH, "all_chunks.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Ingestion completed for all PDFs")
    print(f"📦 Total chunks: {len(all_chunks)}")
    print(f"📁 Saved to {output_file}")


if __name__ == "__main__":
    #run_ingestion_pipeline() # For Chunking and Metadata Inference
    #build_index("data/processed/chunks/all_chunks.json",
     #  "data/processed/index")  # For Indexing the Chunks
    from src.rag.pipeline import RagAssistant

    assistant = RagAssistant("data/processed/index")

    answer, citations, confidence = assistant.ask(
        "HTS Duty Exemption? Ensure to include all relevant sections and cite sources in your answer from User manuals as well as CBP based docs."
    )

    print("\n===== ANSWER =====\n")
    print(answer)

    print("\n===== SOURCES =====\n")
    for c in citations:
        print(
            f"{c['document']} | Section: {c['section']} | Pages: {c['pages']}"
    )
    print("\n===== ANSWER =====\n", answer)
    print("\nConfidence:", confidence)