import chromadb
from pathlib import Path
from src.infrastructure.embeddings.embedder import Embedder
from src.infrastructure.vectorstores.chroma_logic import ChromaManager

def main():
    # 1. Embedder
    embedder = Embedder(
        model_name="BAAI/bge-large-en-v1.5",
        show_progress=False
    )

    # 2. Connect to ChromaDB
    collection = ChromaManager()

    # 3. User Question
    user_query = "What are the common symptoms of abdominal aortic aneurysm?"
    print(f"\nSearching for: {user_query}")

    # 4. Use encode_query to add prefix for BGE model
    query_vector = embedder.encode_query(user_query).tolist()

    # 5. Query top 3
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )

    # 6. Print result
    print("\n--- TOP RESULTS ---")
    for i in range(len(results['documents'][0])):
        content = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        
        # Distance càng nhỏ thì độ tương đồng càng cao (L2 distance)
        print(f"\nResult {i+1} (Score: {distance:.4f}):")
        print(f"Condition: {metadata['condition']}")
        print(f"URL: {metadata['url']}")
        print(f"Content: {content[:200]}...") 
        print("-" * 30)

if __name__ == "__main__":
    main()