from src.retrieval.retriever import Retriever

r = Retriever("data/processed/index")

results = r.query("FDA entry filing procedure")

print("\nRetrieved sections:\n")

for res in results:
    print(res["metadata"].get("section"))
