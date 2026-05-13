import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any

class ChromaManager:
    def __init__(self, db_path: str = "data/vector_db", collection_name: str = "medical_knowledge"):
        self.db_path = db_path
        self.collection_name = collection_name
        
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def upsert_documents(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict], documents: List[str]):
        """Add and update"""
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def query(self, query_embeddings: List[List[float]], n_results: int = 3) -> Dict[str, Any]:
        """Query vector"""
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

    def get_count(self) -> int:
        return self.collection.count()