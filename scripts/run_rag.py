import argparse
import time
from src.application.pipeline_orchestrator import MedicalRAGPipeline

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Medical RAG pipeline")
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="What are the common symptoms of abdominal aortic aneurysm?",
        help="Medical question or symptom description",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top documents to pass to the LLM (default: 3)",
    )
    args = parser.parse_args()

    print("\n Loading pipeline components (embedder, vector DB, reranker, LLM)...")
    t0 = time.perf_counter()
    pipeline = MedicalRAGPipeline()
    print(f"Pipeline ready in {time.perf_counter() - t0:.1f}s\n")

    print(f"{'='*60}")
    print(f"Query: {args.query}")
    print(f"{'='*60}")

    t1 = time.perf_counter()
    result = pipeline.run(args.query, top_n=args.top_n)
    elapsed = time.perf_counter() - t1

    print(f"\n{'─'*60}")
    print(f"LLM Answer  (generated in {elapsed:.2f}s)")
    print(f"{'─'*60}\n")
    print(result["answer"])

    print(f"\n{'─'*60}")
    print(f"Sources ({len(result['sources'])} documents)")
    print(f"{'─'*60}")
    for i, src in enumerate(result["sources"], 1):
        meta = src.get("metadata", {})
        score = src.get("rerank_score", 0)
        print(f"\n  [{i}] {meta.get('condition', '—')} | {meta.get('section', '—')}")
        print(f"      Rerank Score: {score:.4f}")
        print(f"      Preview: {src['content'][:200]}...")

    print(f"\n{'='*60}")
    print(f"Done — total query time: {elapsed:.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
