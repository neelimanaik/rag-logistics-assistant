🚢 RAG Logistics Assistant — Enterprise GenAI Microservice

📌 Overview

The RAG Logistics Assistant is a production-style Retrieval-Augmented Generation (RAG) system designed to support logistics software applications and customs regulatory workflows.

It enables natural-language querying over:
User Manuals (eShip, eFreight, eBrokerage)
Customs & Regulatory Documents (CBP, FDA, OGA guidelines)

The system delivers:

Grounded answers
Section-level citations
Confidence scoring
Guardrail validation
Hybrid semantic + keyword retrieval
REST API deployment
Docker containerization
CI/CD automation

This project reflects enterprise-grade GenAI architecture rather than a tutorial-level implementation.---

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
🏗 Architecture

🌐 High-Level Cloud Architecture

Client (Browser / API Consumer)
            │
            ▼
FastAPI Microservice (Dockerized)
            │
            ▼
Guardrails Layer
   ├─ Domain Validation
   ├─ Out-of-Scope Detection
   └─ Low Confidence Fallback
            │
            ▼
LLM Query Router (Intent Classification)
            │
            ▼
Hybrid Retrieval Engine
   ├─ FAISS Vector Search
   ├─ Keyword Search (BM25-style)
   ├─ Metadata Filtering
   └─ Deduplication
            │
            ▼
Context Assembly
            │
            ▼
LLM Generation (Streaming Enabled)
            │
            ▼
Confidence Scoring
            │
            ▼
Structured JSON Response
   ├─ Answer
   ├─ Citations
   └─ Confidence Level

🔁 DevOps & Deployment Layer

Docker containerization
GitHub Actions CI pipeline
Automated benchmark execution
Swagger API documentation
Production-ready REST endpoint

## Key Features
🧠 Core Features
📄 1. Structure-Aware Document Processing

PDF ingestion
Section-based chunking
Metadata enrichment
Multi-document indexing

🔍 2. Hybrid Retrieval

Combines:

Vector similarity search (Azure OpenAI embeddings + FAISS)
Keyword overlap scoring
Metadata filtering
LLM-based routing

This improves recall for:

Regulatory codes (HTS, OGA, FDA)
Domain-specific terminology
Acronyms and identifiers

🧭 3. LLM Query Routing

LLM-based intent classification dynamically routes queries to:

Functional documentation
Regulatory documentation
Mixed/general cases
Reduces cross-domain contamination.

🛡 4. Guardrails Layer

Domain restriction enforcement
Prompt injection protection
Low-confidence fallback
Out-of-scope query rejection
Prevents hallucinated or unsafe responses.

📊 5. Confidence Scoring

Heuristic confidence based on:
Retrieval similarity
Section agreement
Metadata consistency

Output:

HIGH
MEDIUM
LOW

Used to trigger fallback messaging.

📈 6. Evaluation Framework

Offline benchmark testing:

Precision@K
Section-level grounding validation
Retrieval accuracy tracking
CI pipeline runs automated evaluation.

⚡ 7. Streaming Responses

Token-level streaming improves:

Perceived latency
UX responsiveness
Real-time interaction capability

## Technology Stack

| Layer	               | Technology                  |
| --------------------- | --------------------------- |
| LLM	                  | Azure OpenAI                |
| Embeddings	         | text-embedding-3-large      |
| Vector DB	            | FAISS                       |
| Retrieval	            | Hybrid (Vector + Keyword)   |
| API	                  | FastAPI                     |
| Containerization	   | Docker                      |
| CI/CD	               | GitHub Actions              |
| Evaluation	         | Custom Precision@K          |
| Configuration	      | python-dotenv               |
| PDF Processing	      | LangChain utilities         |

---
🚀 Running the Project

1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Configure Environment

Create .env in project root:

AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_VERSION=...
AZURE_OPENAI_MODEL=...
AZURE_EMBED_MODEL=...

3️⃣ Run API
uvicorn app:app --reload

Swagger available at:

http://127.0.0.1:8000/docs
🐳 Run via Docker

Build image:

docker build -t rag-logistics .

Run container:

docker run -p 8000:8000 rag-logistics
🔄 CI/CD Pipeline

GitHub Actions automatically:

Installs dependencies
Runs evaluation benchmark
Validates pipeline integrity

Workflow location:

.github/workflows/ci.yml
----

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

📌 Production Considerations

Future upgrades may include:

Azure Container Apps deployment
RBAC (role-based document access)
Observability dashboards
Async ingestion pipeline
Automated re-indexing
Vector re-ranking

🎯 Why This Project Matters

This system demonstrates:

End-to-end RAG architecture
Hybrid search implementation
Explainable AI design
Confidence-based hallucination control
DevOps integration
Containerized AI microservices

It reflects enterprise solution architecture thinking, not just experimentation.

📜 License

MIT License

Copyright (c) 2026 Neelima Naik

Permission is hereby granted, free of charge, to any person obtaining a copy...
