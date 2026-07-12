# Running the RAG Assistant locally with Ollama (free, no account)

This runs the whole system on your Windows machine with a local model — no Azure,
no OpenAI key, no cost. Do these in order. Tell me what happens at any step that errors.

## 1. Install Ollama
Download from https://ollama.com/download and run the installer. After install,
Ollama runs a local server at `http://localhost:11434` (it starts automatically).

Verify in a terminal:
```
ollama --version
```

## 2. Pull the two models
- A chat model (answers questions) and an embedding model (turns text into vectors).
```
ollama pull llama3.2
ollama pull nomic-embed-text
```
`llama3.2` is ~2 GB; `nomic-embed-text` is ~0.3 GB. First pull takes a few minutes.

## 3. Set up the Python environment
From the project folder (`rag-logistics-assistant`):
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Create your .env
Copy the example — the defaults already point at Ollama:
```
copy .env.example .env
```

## 5. Build the search index (one time)
This reads the PDFs in `data/raw/`, chunks them, embeds them with Ollama, and
builds the FAISS index:
```
python ingest.py
```
You should see it process the PDFs and print `Index built at: data/processed/index`.

## 6. Run a query — two ways

**A. Quick CLI test**
```
python main.py
```
This asks a sample HTS question and prints the answer + sources.

**B. The API (what an interviewer will want to see)**
```
uvicorn app:app --reload
```
Then open the Swagger UI at http://127.0.0.1:8000/docs and try:
- `POST /ask` — full JSON answer with citations + confidence
- `POST /ask/stream` — the new Server-Sent-Events streaming endpoint (tokens arrive live)

Example body:
```json
{ "question": "What is an HTS duty exemption?" }
```

## Known thing we will tune together
The confidence gate (`src/evaluation/confidence.py`) uses thresholds (0.5 / 1.0)
that were set for a different embedding model. With `nomic-embed-text` the score
scale is different, so you may see **"Insufficient evidence found in documents."**
even for good questions — that's the gate being miscalibrated, not a real failure.

If that happens, run:
```
python debug_scores.py
```
and paste me the scores it prints. We'll set the thresholds from real numbers
(this is exactly the kind of evaluation work the JD asks for).

## If something breaks
Copy me the full error text. Most likely culprits and quick checks:
- `ollama` not running → run `ollama list` to confirm it responds
- model name typo → names are exactly `llama3.2` and `nomic-embed-text`
- import errors → make sure the venv is activated before `pip install`
