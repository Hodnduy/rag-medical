from __future__ import annotations
from typing import Iterable, List, Sequence, Union, overload
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

TextLike = Union[str, Sequence[str]]

# Add prefixes for query embeddings (guide from BGE model docs)
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
        max_seq_length: int | None = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.normalize = normalize
        self.batch_size = batch_size
        self.show_progress = show_progress

        self.model = SentenceTransformer(model_name, device=self.device)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length  # optional

        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode_core(
        self, texts: List[str], normalize: bool
    ) -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=normalize,
            convert_to_tensor=False,
        )
        if not isinstance(embs, np.ndarray):
            embs = np.asarray(embs, dtype=np.float32)
        if embs.dtype != np.float32:
            embs = embs.astype(np.float32, copy=False)
        return embs

    def _resolve_normalize(self, override: bool | None) -> bool:
        return self.normalize if override is None else override

    @overload
    def encode(self, texts: str, *, normalize: bool | None = None,
               batch_size: int | None = None, to_numpy: bool = True) -> np.ndarray: ...
    @overload
    def encode(self, texts: Sequence[str], *, normalize: bool | None = None,
               batch_size: int | None = None, to_numpy: bool = True) -> np.ndarray: ...

    def encode(
        self,
        texts: TextLike,
        *,
        normalize: bool | None = None,
        batch_size: int | None = None,
        to_numpy: bool = True,
    ) -> np.ndarray:
        if isinstance(texts, str):
            batch = [texts]
            single = True
        else:
            batch = list(texts)
            single = False

        if batch_size is not None:
            self.batch_size = batch_size
        norm = self._resolve_normalize(normalize)
        embs = self._encode_core(batch, normalize=norm)
        return embs[0] if single else embs

    def encode_query(self, texts: TextLike, **kw) -> np.ndarray:
        if isinstance(texts, str):
            payload = [_QUERY_PREFIX + texts]
            single = True
        else:
            payload = [_QUERY_PREFIX + t for t in texts]
            single = False
        embs = self._encode_core(payload, normalize=self._resolve_normalize(kw.get("normalize")))
        return embs[0] if single else embs

    def encode_passage(self, texts: TextLike, **kw) -> np.ndarray:
        return self.encode(texts, **kw)

    def encode_iter(
        self,
        texts: Iterable[str],
        *,
        chunk_size: int = 2048,
        normalize: bool | None = None,
    ) -> Iterable[np.ndarray]:
        buffer: List[str] = []
        norm = self._resolve_normalize(normalize)

        def flush(buf: List[str]) -> np.ndarray:
            return self._encode_core(buf, normalize=norm)

        for t in texts:
            buffer.append(t)
            if len(buffer) >= chunk_size:
                yield flush(buffer)
                buffer.clear()
        if buffer:
            yield flush(buffer)

    def similarity(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of vectors.
        A: (m, d), B: (n, d) → (m, n)
        """
        if A.ndim == 1: A = A[None, :]
        if B.ndim == 1: B = B[None, :]
        if self.normalize:
            return A @ B.T
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return A_norm @ B_norm.T

    __call__ = encode


if __name__ == "__main__":
    emb = Embedder(show_progress=True)
    sents = ["This is a red car.", "The sky is blue.","Grass is green.","Roses are red."]
    vecs = emb.encode_passage(sents)
    print("Device:", emb.device)
    print("Dim:", emb.dim)
    print("Shape:", vecs.shape)
    print("Cosine similarity:", float(emb.similarity(vecs[0], vecs[1])))
    q = emb.encode_query("red vehicle")
    print("Query vs passages:", emb.similarity(q, vecs))