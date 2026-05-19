import json
import pickle
from pathlib import Path
import numpy as np
from src.infrastructure.embeddings.embedder import Embedder
from src.infrastructure.vectorstores.chroma_logic import ChromaManager

def main():
    embedder = Embedder(model_name="BAAI/bge-large-en-v1.5", normalize=True, batch_size=32)
    chroma_manager = ChromaManager()

    chunked_file = Path("data/nhs_chunking_cleaned.jsonl")
    if not chunked_file.exists():
        raise FileNotFoundError("Run run_chunking.py first")

    ids, embeddings_text, documents_to_store, metadatas = [], [], [], []
    
    with chunked_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            chunk_data = data["chunk_text"] 
            
            ids.append(chunk_data["id"])
            embeddings_text.append(chunk_data["text"]) 
            documents_to_store.append(chunk_data["text"]) 
            metadatas.append(chunk_data["meta"])

    print(f"Vector Embedding for {len(embeddings_text)} chunks...")
    all_embeddings = []
    for batch_vecs in embedder.encode_iter(embeddings_text, chunk_size=256):
        all_embeddings.append(batch_vecs)
    vectors = np.vstack(all_embeddings).tolist()

    chroma_manager.upsert_documents(
        ids=ids,
        embeddings=vectors,
        metadatas=metadatas,
        documents=documents_to_store
    )
    bm25_corpus = {
        "ids": ids,
        "documents": documents_to_store,
        "metadatas": metadatas
    }
    with open("data/bm25_corpus.pkl", "wb") as f:
        pickle.dump(bm25_corpus, f)
        
    print("Finish")

if __name__ == "__main__":
    main()