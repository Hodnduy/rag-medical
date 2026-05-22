# RAG-Medical — Retrieval-Augmented Generation System For Medical Q&A

A personal project demonstrating a **Retrieval-Augmented Generation (RAG)** pipeline for **medical Q&A** tasks.  
The goal is to build an **end-to-end RAG system** capable of answering medical-related questions using trusted health data sources such as 
[NHS UK](https://www.nhs.uk/conditions/).

You can input your symptoms and the RAG will find the condition which corresponding to your symptoms.
---

## Current Progress

### Phase 1 — Data Crawling
- Implemented a custom web crawler to collect structured medical data from NHS conditions pages.
- Extracted and stored data with fields:
  ```json
  {
    "condition": "...",
    "section": "Symptoms",
    "url": "https://www.nhs.uk/conditions/...",
    "content": "Symptoms of ..."
  }
  ```
  
### Phase 2 — Data Cleaning & Chunking

After crawling, the raw JSON data from `nhs_conditions.jsonl` needs to be **cleaned and normalized** before being chunking.

#### Data Cleaning
Use the provided notebook to clean and standardize text:
Just run all cell in `notebooks/eda_preprocessing.ipynb`. It create cleaned file `nhs_cleaned.jsonl`

#### Chunking
Use `scripts/run_chunking.py` to chunking data with this command:
```bash
python -m scripts.run_chunking --in-dir data/ --out-dir data/chunk_data
```

### Phase 3 — Vector Storage & Embedding

In this phase, the cleaned chunks are transformed into high-dimensional vectors and stored in a vector database for efficient retrieval.

- **Embedding Model**: Using `BAAI/bge-large-en-v1.5` (Top-tier on MTEB leaderboard) with custom query prefixes.
- **Vector Store**: Implemented **ChromaDB** with a persistent local backend.
- **Architecture**: Decoupled infrastructure logic into `ChromaManager` to ensure modularity.

#### How to run Embedding:
```bash
# Ensure you have cleaned data in data/
python -m scripts.run_embedding
```
#### How to test Retrieval & Reranking:
The retrieval process involves an **Early Exit Distance Gate** to filter out irrelevant queries, followed by a **Cross-Encoder Reranker** (`BAAI/bge-reranker-v2-m3`) to accurately re-order the retrieved candidates.

```bash
python -m scripts.query
```

### Phase 4 — LLM Generation (The "G" in RAG)

The final piece of the RAG pipeline: a Large Language Model generates answers grounded in the retrieved context.

- **LLM**: `Qwen/Qwen3-4B-Instruct-2507` served locally via **vLLM** (`http://127.0.0.1:8000/v1`).
- **Prompt Engineering**: System prompt with strict anti-hallucination rules — the model is forbidden from fabricating information outside the provided documents.
- **Architecture**: `MedicalLLMGenerator` implements the `LLMGenerator` ABC (domain layer) and reads all config from a unified `Settings` (pydantic).
- **Full Pipeline**: `MedicalRAGPipeline.run()` now orchestrates: **Query → Embed → Hybrid Search → Rerank → Generate**.

#### How to run the full RAG pipeline:
```bash
# Run local LLM first
vllm-metal --model Qwen/Qwen3-4B-Instruct-2507 --port 8000

# Make sure vLLM is running with Qwen3-4B
python -m scripts.run_rag --query "What are the symptoms of diabetes?"

# Or with custom top_n
python -m scripts.run_rag --query "I have a headache and fever" --top-n 5
```

---

## Next Steps (To-Do)

### Phase 5 — Backend API & Services (FastAPI)
- **Objective**: Serve the RAG system as a REST API.
- **Tasks**:
  - Set up FastAPI with routes (e.g., `POST /api/chat`).
  - Implement request validation, error handling, and structured JSON responses.
  - Setup dependency injection for the Embedder, ChromaManager, Reranker, and Generator.

### Phase 6 — User Interface (UI)
- **Objective**: Build a user-friendly frontend to interact with the Medical Chatbot.
- **Tasks**:
  - Use **Streamlit** or **Gradio** for a quick conversational UI prototype.
  - Alternatively, build a modern Web App using **React/Next.js** for a premium chat experience.

### Phase 7 — Evaluation & Observability
- **Objective**: Measure the quality of answers and monitor the system.
- **Tasks**:
  - Implement RAG Evaluation metrics (Faithfulness, Answer Relevance, Context Precision) using tools like Ragas or TruLens.
  - Add observability logs and tracing using Langfuse or Phoenix.