"""Print retrieval scores for a sample query.

Used to calibrate the confidence thresholds in src/evaluation/confidence.py for
whichever embedding model you are running, instead of guessing.
"""

from src.config.settings import settings
from src.retrieval.retriever import Retriever

if __name__ == "__main__":
    settings.validate()
    r = Retriever(settings.index_path)
    query = "HTS duty exemption"
    results = r.query(query, k=10)
    print(f"Query: {query}\n")
    for res in results:
        print(f"  score={res.get('score')}   section={res['metadata'].get('section')}")
