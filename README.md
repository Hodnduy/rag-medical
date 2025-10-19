# 🧠 RAG-Medical — Local Retrieval-Augmented Generation System

A personal AI/ML project demonstrating a **Retrieval-Augmented Generation (RAG)** pipeline for **medical Q&A** tasks.  
The goal is to build an **end-to-end RAG system** capable of answering medical-related questions using trusted health data sources such as 
[NHS UK](https://www.nhs.uk/conditions/).

You can input your symptoms and the RAG will find the condition which corresponding to your symptoms.
---

## 🚀 Current Progress

### ✅ Phase 1 — Data Crawling & Cleaning (Completed)
- Implemented a custom web crawler to collect structured medical data from NHS conditions pages.
- Extracted and stored data with fields:
  ```json
  {
    "condition": "...",
    "section": "Symptoms",
    "url": "https://www.nhs.uk/conditions/...",
    "content": "Symptoms of ..."
  }
