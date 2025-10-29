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
python -m scripts.run_chunking --in data/ --out data/chunk_data
```