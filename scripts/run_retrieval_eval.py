"""
Retrieval evaluation script for Medical RAG.

Evaluates retrieval quality independently of LLM generation across 4 configurations:
1. Dense only (ChromaDB)
2. Sparse only (BM25)
3. Hybrid (Dense + Sparse)
4. Hybrid + Rerank

Metrics computed:
- Hit@K: Fraction of queries where at least one retrieved document is relevant
- MRR@K: Mean Reciprocal Rank (1/rank of first hit, averaged over queries)
- Latency (Avg and P95 in milliseconds)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.application.pipeline_orchestrator import MedicalRAGPipeline


STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
    'not', 'no', 'nor', 'so', 'yet', 'both', 'either', 'neither', 'each',
    'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such',
    'than', 'too', 'very', 'just', 'also', 'that', 'this', 'these', 'those',
    'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your',
    'he', 'him', 'his', 'she', 'her', 'who', 'which', 'what', 'when', 'where',
    'how', 'if', 'then', 'about', 'up', 'out', 'down', 'over', 'under'
}


def tokenize(text: str) -> List[str]:
    """Simple word tokenizer: lowercase, remove punctuation, filter stopwords."""
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    return [w for w in words if w and w not in STOPWORDS]


def is_hit(doc_text: str, reference: str, threshold: float = 0.3) -> bool:
    """Check if a retrieved document is relevant based on keyword overlap with reference."""
    ref_tokens = set(tokenize(reference))
    if not ref_tokens:
        return False
    doc_tokens = set(tokenize(doc_text))
    overlap = len(ref_tokens & doc_tokens)
    return (overlap / len(ref_tokens)) >= threshold


def compute_metrics(results: List[Dict[str, Any]], k_values: List[int]) -> Dict[str, float]:
    """
    Compute Hit@K and MRR@K metrics for retrieval results.

    Args:
        results: List of dicts with 'query', 'reference', and 'retrieved_docs'.
        k_values: List of K cutoffs (e.g., [3, 5]).

    Returns:
        dict: Mapping of metric keys like 'hit@3', 'mrr@3', etc. to float values.
    """
    metrics = {}
    n_queries = len(results)

    if n_queries == 0:
        for k in k_values:
            metrics[f"hit@{k}"] = 0.0
            metrics[f"mrr@{k}"] = 0.0
        return metrics

    for k in k_values:
        total_hits = 0
        total_rr = 0.0

        for item in results:
            reference = item.get("reference", "")
            docs = item.get("retrieved_docs", [])[:k]

            hit_found = False
            for rank, doc in enumerate(docs, start=1):
                if is_hit(doc, reference):
                    total_rr += 1.0 / rank
                    hit_found = True
                    break

            if hit_found:
                total_hits += 1

        metrics[f"hit@{k}"] = round(total_hits / n_queries, 4)
        metrics[f"mrr@{k}"] = round(total_rr / n_queries, 4)

    return metrics


def retrieve_dense_only(pipeline: MedicalRAGPipeline, query: str, k: int) -> List[str]:
    """Retrieve top-K documents using dense retrieval (ChromaDB) only."""
    result = pipeline._run_dense_retrieval(query)
    docs = []
    if result.get('documents') and len(result['documents']) > 0 and result['documents'][0]:
        docs = result['documents'][0][:k]
    return docs


def retrieve_sparse_only(pipeline: MedicalRAGPipeline, query: str, k: int) -> List[str]:
    """Retrieve top-K documents using sparse retrieval (BM25) only."""
    indices = pipeline._run_sparse_retrieval(query)
    docs = [pipeline.bm25_data['documents'][i] for i in indices[:k]]
    return docs


def retrieve_hybrid(pipeline: MedicalRAGPipeline, query: str, k: int) -> List[str]:
    """Retrieve top-K documents using hybrid retrieval (Dense + Sparse)."""
    candidates = pipeline.search_hybrid(query)
    return [c['content'] for c in candidates[:k]]


def retrieve_hybrid_rerank(pipeline: MedicalRAGPipeline, query: str, k: int) -> List[str]:
    """Retrieve top-K documents using hybrid retrieval with cross-encoder reranking."""
    candidates = pipeline.search_hybrid(query)
    if not candidates:
        return []
    # Only rerank top-10 candidates (RRF already placed best docs first)
    rerank_pool = candidates[:max(10, k)]
    reranked = pipeline.reranker.rerank(query, rerank_pool, top_n=k)
    return [c['content'] for c in reranked]


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality across different modes")
    parser.add_argument("--dataset", type=str, default="data/eval_dataset.json", help="Path to evaluation dataset JSON")
    parser.add_argument("--out-file", type=str, default="data/retrieval_eval_results.json", help="Path to output JSON")
    parser.add_argument("--max-k", type=int, default=5, help="Maximum K for retrieval")
    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Filter out out-of-domain questions (those whose reference indicates info is not available)
    # Also support both 'user_input' / 'reference' and 'question' / 'ground_truth' key conventions
    eval_items = []
    for item in dataset:
        query = item.get("user_input") or item.get("question", "")
        reference = item.get("reference") or item.get("ground_truth", "")
        ref_lower = reference.lower()

        if (
            "not contain" not in ref_lower
            and "cannot be answered" not in ref_lower
            and "not available" not in ref_lower
        ):
            eval_items.append({
                "user_input": query,
                "reference": reference
            })

    print(f"Loaded {len(eval_items)} in-domain questions (filtered from {len(dataset)} total)")
    if not eval_items:
        print("No evaluation items to evaluate. Exiting.")
        return

    print("Initializing pipeline...")
    pipeline = MedicalRAGPipeline()

    modes: Dict[str, Callable[[MedicalRAGPipeline, str, int], List[str]]] = {
        "Dense Only": retrieve_dense_only,
        "Sparse Only (BM25)": retrieve_sparse_only,
        "Hybrid": retrieve_hybrid,
        "Hybrid + Rerank": retrieve_hybrid_rerank,
    }

    k_values = [3, 5]
    if args.max_k not in k_values:
        k_values.append(args.max_k)
        k_values.sort()

    max_k = max(k_values)
    all_results = {}

    for mode_name, retrieve_fn in modes.items():
        print(f"\nEvaluating: {mode_name}...")
        mode_results = []
        latencies = []

        for item in eval_items:
            query = item["user_input"]
            reference = item["reference"]

            t0 = time.perf_counter()
            docs = retrieve_fn(pipeline, query, max_k)
            latency = time.perf_counter() - t0
            latencies.append(latency)

            mode_results.append({
                "query": query,
                "reference": reference,
                "retrieved_docs": docs
            })

        metrics = compute_metrics(mode_results, k_values)

        if latencies:
            metrics["avg_latency_ms"] = round(sum(latencies) / len(latencies) * 1000, 1)
            p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
            metrics["p95_latency_ms"] = round(sorted(latencies)[p95_idx] * 1000, 1)
        else:
            metrics["avg_latency_ms"] = 0.0
            metrics["p95_latency_ms"] = 0.0

        all_results[mode_name] = metrics

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Mode':<25} {'Hit@3':>8} {'Hit@5':>8} {'MRR@3':>8} {'MRR@5':>8} {'Avg(ms)':>10} {'P95(ms)':>10}")
    print("-" * 80)
    for mode, m in all_results.items():
        print(
            f"{mode:<25} "
            f"{m['hit@3']:>8.3f} "
            f"{m['hit@5']:>8.3f} "
            f"{m['mrr@3']:>8.3f} "
            f"{m['mrr@5']:>8.3f} "
            f"{m['avg_latency_ms']:>10.1f} "
            f"{m['p95_latency_ms']:>10.1f}"
        )
    print("=" * 80)

    # Save results
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.out_file}")


if __name__ == "__main__":
    main()
