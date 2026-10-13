import torch
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any


def _auto_device() -> str:
    """Pick the best available accelerator: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MedicalReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 16,
    ):
        """
        Create Cross-Encoder for reranking.

        Args:
            model_name: HuggingFace model ID for the cross-encoder.
                        Default changed from ``bge-reranker-v2-m3`` (560M) to
                        ``bge-reranker-base`` (110M) for ~5× faster inference
                        with comparable accuracy on English medical text.
            device:     Compute device ("cuda", "mps", "cpu"). Auto-detected
                        if not specified.
            batch_size: Batch size for cross-encoder inference.
        """
        self.device = device or _auto_device()
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=self.device)

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
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # 3. Score the result
        for i, score in enumerate(scores):
            candidates[i]['rerank_score'] = float(score)

        # 4. Sort candidate
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

        return candidates[:top_n]