# RAG Logistics Assistant — Enterprise-Grade GenAI System

## Overview

This project implements an end-to-end Retrieval-Augmented Generation (RAG) system designed for logistics software environments where users and developers must navigate complex documentation such as:

* Application user manuals (eShip, eFreight, eBrokerage)
* Customs regulations and compliance documents

The assistant enables natural-language querying over domain documents, returning grounded answers with traceable citations, confidence scoring, and hybrid retrieval.

This system mirrors production enterprise architecture rather than tutorial-level implementations.

---

## Problem Statement

Logistics application users and developers frequently consult multiple documents to complete operational or regulatory tasks. This results in:

* Time delays for critical shipping workflows
* Support team overload
* Risk of regulatory misinterpretation

This project demonstrates how a RAG system can:

* Provide instant contextual answers
* Reduce dependency on support teams
* Improve compliance navigation
* Scale across document types

---

## Key Features

### Document Processing

* Structure-aware PDF ingestion
* Section-based chunking
* Metadata enrichment
* Multi-document indexing

### Retrieval Architecture

* Vector embeddings (Azure OpenAI)
* FAISS vector index
* Hybrid retrieval

  * Semantic similarity
  * Keyword overlap scoring
* Metadata-based filtering
* LLM-driven query routing

### Answer Generation

* Context-grounded responses
* Source citations
* Confidence scoring
* Streaming token output

---

## System Architecture

```
PDF Docs
   │
   ▼
Structure Chunking
   │
   ▼
Embedding Generation
   │
   ▼
FAISS Vector Store
   │
   ▼
Hybrid Retrieval Engine
   │
   ├─ Metadata Filtering
   ├─ LLM Query Routing
   └─ Confidence Scoring
   ▼
Grounded LLM Response
   │
   ▼
Streaming Answer + Citations
```

---

## Technology Stack

| Layer          | Tools                  |
| -------------- | ---------------------- |
| LLM            | Azure OpenAI           |
| Embeddings     | text-embedding-3-large |
| Vector DB      | FAISS                  |
| Orchestration  | Python                 |
| PDF Processing | LangChain utilities    |
| Similarity     | NumPy                  |
| Configuration  | python-dotenv          |

---

## Design Decisions & Tradeoffs

### FAISS vs Managed Vector DB

**Chosen:** FAISS

* Lightweight
* Offline capability
* Fast local iteration

**Tradeoff:**

* No distributed scaling
* No managed indexing

**Production Alternative:**
Azure AI Search / Pinecone / Elastic

---

### Hybrid Retrieval vs Pure Vector

**Chosen:** Hybrid

* Improves recall for domain identifiers
* Captures semantic + lexical matches

**Tradeoff:**

* Slight complexity increase
* Score fusion tuning required

---

### LLM Routing

**Chosen:** Intent classification via LLM

**Benefits**

* Flexible domain scaling
* Reduced rule maintenance

**Tradeoff**

* Additional LLM call latency

---

## Evaluation Signals

Example observed metrics:

* Compression Ratio: ~10%
* Coverage Score: ~50%
* Hybrid retrieval accuracy improvement vs vector-only
* Context-grounded answer traceability

---

## Running the Project

### Install

```bash
pip install -r requirements.txt
```

### Configure

Create `.env` in project root:

```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_MODEL=...
AZURE_EMBED_MODEL=...
```

### Run

```bash
python -m main
```

---

## Future Enhancements

* UI chat interface (FastAPI/Streamlit)
* Role-based access control
* Vector re-ranking
* Async ingestion pipeline
* Feedback learning loop
* Observability dashboard

---

## Author Perspective

This project was developed to demonstrate architectural thinking around enterprise GenAI systems including:

* Retrieval reliability
* Explainability
* Modular scalability
* Production tradeoff awareness

---

## License


