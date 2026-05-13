from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class MedicalReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Create Cross-Encoder. 
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Return top N answers
        """
        if not candidates:
            return []

        # 1. Pair (Question, Doc)
        passages = [c['content'] for c in candidates]
        pairs = [[query, p] for p in passages]
        
        # 2. Predict Relevance Score
        scores = self.model.predict(pairs)
        
        # 3. Score the result
        for i, score in enumerate(scores):
            candidates[i]['rerank_score'] = float(score)
            
        # 4. Sort candidate
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return candidates[:top_n]