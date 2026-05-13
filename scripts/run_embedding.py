import json
import uuid
import numpy as np
from pathlib import Path
import chromadb
from chromadb.config import Settings
from src.infrastructure.embeddings.embedder import Embedder
from src.infrastructure.vectorstores.chroma_logic import ChromaManager

def load_medical_json(file_path: str):
    """Read JSON file"""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            # Context Enrichment
            input_text = f"Condition: {data['condition']}. Section: {data['section']}. Content: {data['content']}"
            
            documents.append({
                "id": str(uuid.uuid4()),
                "text": input_text,
                "metadata": {
                    "condition": data['condition'],
                    "section": data['section'],
                    "url": data['url'],
                    "original_content": data['content']
                }
            })
    return documents

def main():
    # 1. Create Embedder
    embedder = Embedder(
        model_name="BAAI/bge-large-en-v1.5",
        normalize=True,
        batch_size=32,
        show_progress=True
    )

    # 2. Create ChromaDB Client
    collection = ChromaManager()

    # 3. Load data
    data_file = "data/nhs_cleaned.jsonl"
    processed_docs = load_medical_json(data_file)
    
    texts_to_embed = [d["text"] for d in processed_docs]
    metadatas = [d["metadata"] for d in processed_docs]
    ids = [d["id"] for d in processed_docs]

    # 4. Batch Embedding
    print(f"Processing embedding for {len(texts_to_embed)} doc")
    
    all_embeddings = []
    # Use encode_iter in class Embedder
    for batch_vecs in embedder.encode_iter(texts_to_embed, chunk_size=256):
        all_embeddings.append(batch_vecs)

    vectors = np.vstack(all_embeddings).tolist()

    # 5. Push data to VectorDB
    collection.add(
        embeddings=vectors,
        metadatas=metadatas,
        documents=[d["metadata"]["original_content"] for d in processed_docs],
        ids=ids
    )

if __name__ == "__main__":
    main()