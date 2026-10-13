import pickle
import re
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from rank_bm25 import BM25Okapi
from src.infrastructure.embeddings.embedder import Embedder
from src.infrastructure.vectorstores.chroma_logic import ChromaManager
from src.infrastructure.rerank.reranker import MedicalReranker
from src.infrastructure.llm.llm_generator import MedicalLLMGenerator


class MedicalRAGPipeline:
    """
    Full RAG pipeline:
        Query → Embed → Hybrid Search (Dense + Sparse + RRF)
              → Rerank → **LLM Generate**
    """

    # Default RRF constant (commonly k=60 in literature)
    RRF_K = 60

    def __init__(self, top_k_dense: int = 15, top_k_sparse: int = 15):
        self.embedder = Embedder(show_progress=False)
        self.chroma_manager = ChromaManager()
        self.reranker = MedicalReranker()

        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse

        bm25_path = Path("data/bm25_corpus.pkl")
        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 corpus file not found at: {bm25_path}")

        with open(bm25_path, "rb") as f:
            self.bm25_data = pickle.load(f)

        tokenized_corpus = [self._tokenize(doc) for doc in self.bm25_data["documents"]]
        self.bm25 = BM25Okapi(tokenized_corpus)


        self.generator = MedicalLLMGenerator()

    def _tokenize(self, text: str) -> List[str]:
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        return [word for word in clean_text.split(" ") if word]

    def _run_dense_retrieval(self, query: str) -> Dict[str, Any]:
        """Dense search in ChromaDB"""
        try:
            query_vector = self.embedder.encode_query(query).tolist()
            return self.chroma_manager.query(query_embeddings=[query_vector], n_results=self.top_k_dense)
        except Exception as e:
            print(f"[Error] Dense retrieval failed: {e}")
            return {'documents': [[]], 'ids': [[]], 'metadatas': [[]], 'distances': [[]]}

    def _run_sparse_retrieval(self, query: str) -> List[int]:
        """Search BM25"""
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        return sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:self.top_k_sparse]

    def search_hybrid(self, query: str) -> List[Dict[str, Any]]:
        """
        Parallel Dense + Sparse search, merged via Reciprocal Rank Fusion (RRF).

        RRF score for document d:
            score(d) = 1/(k + rank_dense(d)) + 1/(k + rank_sparse(d))
        where k = RRF_K (default 60).  Documents appearing in only one list
        receive a single RRF contribution; the missing term defaults to 0.

        Results are sorted by descending RRF score so the caller always gets
        the most relevant candidates first.
        """
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(self._run_dense_retrieval, query)
            future_sparse = executor.submit(self._run_sparse_retrieval, query)

            dense_results = future_dense.result()
            top_subset_idx = future_sparse.result()

        k = self.RRF_K
        candidate_pool: Dict[str, Dict[str, Any]] = {}

        # --- Dense results with RRF contribution ---
        if dense_results['documents'] and dense_results['documents'][0]:
            for rank, i in enumerate(range(len(dense_results['documents'][0]))):
                doc_id = dense_results['ids'][0][i]
                candidate_pool[doc_id] = {
                    "content": dense_results['documents'][0][i],
                    "metadata": dense_results['metadatas'][0][i],
                    "dense_distance": dense_results['distances'][0][i],
                    "rrf_score": 1.0 / (k + rank + 1),   # rank is 0-indexed, +1 for 1-indexed
                }

        # --- Sparse results with RRF contribution ---
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)

        for rank, idx in enumerate(top_subset_idx):
            if doc_scores[idx] <= 0:
                continue
            doc_id = self.bm25_data["ids"][idx]
            rrf_contribution = 1.0 / (k + rank + 1)

            if doc_id in candidate_pool:
                # Document found in both lists — accumulate RRF
                candidate_pool[doc_id]["rrf_score"] += rrf_contribution
            else:
                candidate_pool[doc_id] = {
                    "content": self.bm25_data["documents"][idx],
                    "metadata": self.bm25_data["metadatas"][idx],
                    "dense_distance": 999.0,
                    "rrf_score": rrf_contribution,
                }

        # Sort by RRF score (highest first)
        candidates = list(candidate_pool.values())
        candidates.sort(key=lambda c: c["rrf_score"], reverse=True)
        return candidates


    def run(self, user_query: str, top_n: int = 3) -> Dict[str, Any]:
        """
        Full RAG pipeline:
            Hybrid Search → Reranking → LLM Generation

        Returns:
            dict with keys:
                - query:     the original user query
                - answer:    the LLM-generated answer
                - sources:   the top-N reranked context documents used
        """
        # 1. Hybrid retrieval
        candidates = self.search_hybrid(user_query)

        if not candidates:
            return {
                "query": user_query,
                "answer": "I could not find any relevant medical information in the database for your query.",
                "sources": [],
            }

        # 2. Rerank — send only the top candidates to reduce latency
        #    (RRF already placed the most promising docs first)
        rerank_pool_size = min(len(candidates), max(10, top_n))
        top_docs = self.reranker.rerank(user_query, candidates[:rerank_pool_size], top_n=top_n)

        # 3. Generate answer via LLM
        answer = self.generator.generate_answer(user_query, top_docs)

        return {
            "query": user_query,
            "answer": answer,
            "sources": top_docs,
        }