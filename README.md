# RAG-Medical — Retrieval-Augmented Generation System for Medical Q&A

> A **Retrieval-Augmented Generation (RAG)** pipeline designed for medical Q&A tasks. Users can input symptoms or medical questions, and the system retrieves trusted health data sourced from [NHS UK](https://www.nhs.uk/conditions/) to generate accurate, grounded answers.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Running the Data Pipeline](#running-the-data-pipeline)
   - [Step 1 — Web Crawling](#step-1--web-crawling)
   - [Step 2 — Data Cleaning](#step-2--data-cleaning)
   - [Step 3 — Chunking](#step-3--chunking)
   - [Step 4 — Embedding & Vector Storage](#step-4--embedding--vector-storage)
4. [Running the RAG Pipeline](#running-the-rag-pipeline)
   - [Start the LLM Server](#start-the-llm-server)
   - [Run RAG via CLI](#run-rag-via-cli)
   - [Run RAG via API](#run-rag-via-api)
5. [Makefile Commands](#makefile-commands)

---

## System Requirements

| Component       | Requirement / Version                  |
|-----------------|----------------------------------------|
| **Python**      | 3.12                                   |
| **uv**          | >= 0.7 (Package Manager)               |
| **OS**          | macOS (Apple Silicon) — for vLLM Metal |
| **RAM**         | >= 16 GB (recommended for local LLM)   |
| **Disk Space**  | >= 10 GB (model weights + vector DB)   |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hodnduy/rag-medical
cd rag-medical
```

### 2. Install `uv` (if not already installed)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew
brew install uv
```

### 3. Install dependencies

```bash
# Set up a virtual environment and install all dependencies
make install
# Or run directly:
uv sync
```

This will automatically create a `.venv/` directory and install all required packages from `pyproject.toml`.

---

## Running the Data Pipeline

> **Note:** If you already have data populated in `data/` and want to test the interactive pipeline, you can proceed directly to [Running the RAG Pipeline](#running-the-rag-pipeline).

### Step 1 — Web Crawling

Fetch raw medical descriptions from NHS UK:

```bash
make crawl
# Or run directly:
uv run python scripts/ingest_nhs.py
```

**Output:** `data/nhs_conditions.jsonl` — containing fields like `condition`, `section`, `url`, and `content`.

### Step 2 — Data Cleaning

Open and run all cells inside the following Jupyter notebook:

```
notebooks/eda_preprocessing.ipynb
```

**Output:** `data/nhs_cleaned.jsonl` — cleaned, standard text corpus without web artifacts.

### Step 3 — Chunking

Split the cleaned documents into uniform chunks:

```bash
make chunk
# Or run manually with custom arguments:
uv run python scripts/run_chunking.py \
    --in-dir data/ \
    --out-dir data/ \
    --max-words 530 \
    --overlap-words 80 \
    --min-words 40
```

| Parameter          | Default | Description                                  |
|--------------------|---------|----------------------------------------------|
| `--max-words`      | 530     | Maximum word count per chunk                 |
| `--overlap-words`  | 80      | Word overlap between consecutive chunks       |
| `--min-words`      | 40      | Discard any chunk with fewer than X words    |

**Output:** `data/nhs_chunking.jsonl`

### Step 4 — Embedding & Vector Storage

Embed the chunks and ingest them into ChromaDB:

```bash
make embed
# Or run directly:
uv run python scripts/run_embedding.py
```

- **Embedding Model:** `BAAI/bge-large-en-v1.5`
- **Vector Database:** ChromaDB (stored locally under `data/vector_db/`)
- **Sparse Indexing:** `data/bm25_corpus.pkl` (for hybrid search)

> The first execution will download the embedding model weights (~1.3 GB). Subsequent runs will be loaded from local cache.

### Run All Pipeline Steps Automatically

```bash
make pipeline
```

This runs `crawl` → `chunk` → `embed` sequentially.

---

## Running the RAG Pipeline

### Start the LLM Server

The pipeline uses **vLLM** to serve `Qwen/Qwen3-4B-Instruct-2507` locally. Start the server before running the query commands:

```bash
# On macOS Apple Silicon (using vLLM Metal)
vllm-metal --model Qwen/Qwen3-4B-Instruct-2507 --port 8000
```

> The first time you serve this model, it will download weights (~8 GB). The server is ready once you see: `Uvicorn running on http://127.0.0.1:8000`.

To verify your local model endpoint is reachable:

```bash
uv run python scripts/test_model.py
```

### Run RAG via CLI

Once the server is running, open a **new terminal window** and query the system:

```bash
# Run with default query
make rag

# Run with custom query
make rag QUERY="I have a headache and fever"

# Adjust the number of retrieved context documents
make rag QUERY="What are the symptoms of diabetes?" TOP_N=5

# Or run using Python directly
uv run python scripts/run_rag.py --query "What are the symptoms of diabetes?" --top-n 5
```

**Example Output:**

```
============================================================
Query: What are the symptoms of diabetes?
============================================================

──────────────────────────────────────────────────────────────
LLM Answer  (generated in 3.25s)
──────────────────────────────────────────────────────────────

Based on the provided medical documents, the common symptoms of diabetes include...

──────────────────────────────────────────────────────────────
Sources (3 documents)
──────────────────────────────────────────────────────────────

  [1] Type 2 diabetes | Symptoms
      Rerank Score: 0.9823
      Preview: The main symptoms of diabetes include...
```

### Run RAG via API

Start the FastAPI application:

```bash
uv run python scripts/run_api.py
```

The server listens on `http://localhost:8080`. Query the endpoint via curl:

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of diabetes?"}'
```

---

## 5. Evaluation

The project provides a two-tier evaluation framework:

1. **Retrieval Ablation Evaluation** (`make retrieval-eval`): Evaluates retrieval quality independently of the LLM across 4 modes (Dense, BM25, Hybrid, Hybrid + Rerank) measuring **Hit@K**, **MRR@K**, and **Latency**.
2. **End-to-End RAG Evaluation** (`make evaluate`): Evaluates the complete pipeline using the **Ragas 0.4.x** framework across 4 RAG Triad metrics (**Faithfulness**, **Answer Relevancy**, **Context Precision**, **Context Recall**).

### Setup

Ensure your Gemini API key is configured in `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Evaluation Commands

| Command | Script | Description | Outputs |
| :--- | :--- | :--- | :--- |
| `make retrieval-eval` | `scripts/run_retrieval_eval.py` | Benchmarks Dense vs BM25 vs Hybrid vs Reranker | `data/retrieval_eval_results.json` |
| `make evaluate` | `scripts/run_evaluation.py` | Full Ragas 0.4.x RAG Triad evaluation | `data/eval_results.json`<br>`data/eval_summary.json` |
| `make generate-testset` | `scripts/generate_testset.py` | Synthesises test cases using Gemini from NHS data | `data/eval_testset.json` |
| `make eval-all` | — | Runs both `evaluate` and `retrieval-eval` sequentially | All of the above |

### Evaluation Dataset

The default evaluation dataset ([`data/eval_dataset.json`](data/eval_dataset.json)) contains **50 curated clinical questions** structured into 4 distinct categories:
- **Simple Factual (15 questions)**: Single-condition symptoms and standard treatments.
- **Comparative & Multi-hop (15 questions)**: Differential symptoms (e.g. Type 1 vs Type 2 diabetes, Crohn's vs Ulcerative colitis).
- **Out-of-domain / Negative (10 questions)**: Non-NHS / unanswerable queries to test model refusal capability.
- **Emergency / Critical (10 questions)**: Life-threatening emergencies (stroke, heart attack, sepsis, anaphylaxis) requiring emergency dispatch (999) instructions.

---

## Makefile Commands Reference

| Command                 | Description                                                  |
|-------------------------|--------------------------------------------------------------|
| `make install`          | Installs dependencies using `uv sync`                        |
| `make crawl`            | Ingests raw data from NHS UK website                         |
| `make chunk`            | Segments raw JSONL data into chunk files                     |
| `make embed`            | Computes embeddings and writes database collections          |
| `make pipeline`         | Sequence execution of crawl, chunk, and embed stages         |
| `make rag`              | Runs the terminal-based query agent                          |
| `make test-model`       | Asserts correct local model availability                     |
| `make retrieval-eval`   | Evaluates Dense vs BM25 vs Hybrid vs Rerank (Hit@K, MRR@K)   |
| `make evaluate`         | Runs full RAG Triad evaluation using Ragas 0.4.x & Gemini    |
| `make eval-all`         | Runs both retrieval and end-to-end evaluation                |
| `make generate-testset` | Generates synthetic testset using Gemini from NHS documents  |
| `make clean-cache`      | Sweeps away intermediate caches (`__pycache__`, `.pyc`)      |