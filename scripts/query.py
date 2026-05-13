import chromadb
from pathlib import Path
from src.infrastructure.embeddings.embedder import Embedder
from src.infrastructure.vectorstores.chroma_logic import ChromaManager
from src.infrastructure.rerank.reranker import MedicalReranker

def main():
    embedder = Embedder(show_progress=False)
    collection = ChromaManager()
    reranker = MedicalReranker()

    user_query = "What are the common symptoms of abdominal aortic aneurysm?"
    print(f"\nSearching for: {user_query}")

    query_vector = embedder.encode_query(user_query).tolist()

    query_vector = embedder.encode_query(user_query).tolist()
    raw_results = collection.query(query_embeddings=[query_vector], n_results=10)

    if not raw_results['distances'] or not raw_results['distances'][0]:
        print(">>> Answer: My database is currently empty.")
        return

    # EARLY EXIT (DISTANCE GATE)
    best_distance = raw_results['distances'][0][0]
    DISTANCE_THRESHOLD = 1.0 

    if best_distance > DISTANCE_THRESHOLD:
        print(f"\n[Fast Exit] Best Distance ({best_distance:.4f}) exceeds threshold ({DISTANCE_THRESHOLD}).")
        print(">>> Answer: I'm sorry, I don't have information on this topic in database.")
        return

    candidates = []
    for i in range(len(raw_results['documents'][0])):
        candidates.append({
            "content": raw_results['documents'][0][i],
            "metadata": raw_results['metadatas'][0][i],
            "distance": raw_results['distances'][0][i]
        })

    print(f"Reranking candidates...")
    final_results = reranker.rerank(user_query, candidates, top_n=3)

    print("\n--- TOP RESULTS ---")
    for i, res in enumerate(final_results):
        print(f"\nRank {i+1} | Rerank Score: {res['rerank_score']:.4f} | Vector Distance: {res['distance']:.4f}")
        print(f"Condition: {res['metadata']['condition']}")
        print(f"Content: {res['content'][:200]}...")
        print("-" * 40)

if __name__ == "__main__":
    main()