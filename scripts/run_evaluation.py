"""
RAG Evaluation Script using Ragas 0.4.x and Gemini API.

Evaluates the full RAG pipeline (Retrieval + Generation) across 4 metrics:
  - Faithfulness:       Does the answer stay faithful to the retrieved contexts?
  - Answer Relevancy:   Is the answer relevant to the question?
  - Context Precision:  Are the retrieved contexts precise w.r.t. the ground truth?
  - Context Recall:     Do the retrieved contexts cover the ground truth?
"""

import json
import os
import sys
import types
import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Patch for ragas 0.4.x: stub out missing langchain_community.chat_models.vertexai
# ---------------------------------------------------------------------------
try:
    import langchain_community.chat_models
except ImportError:
    pass
sys.modules["langchain_community.chat_models.vertexai"] = types.ModuleType(
    "langchain_community.chat_models.vertexai"
)
sys.modules["langchain_community.chat_models.vertexai"].ChatVertexAI = type(
    "ChatVertexAI", (), {}
)

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ---------------------------------------------------------------------------
# Import the local RAG pipeline
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.application.pipeline_orchestrator import MedicalRAGPipeline

    _HAS_RAG = True
except ImportError:
    _HAS_RAG = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset(path: str) -> list[dict]:
    """Load the eval dataset and normalise keys to Ragas 0.4.x names."""
    with open(path, "r") as f:
        raw = json.load(f)

    items = []
    for entry in raw:
        items.append(
            {
                # Support both legacy ("question"/"ground_truth") and new keys
                "user_input": entry.get("user_input") or entry.get("question", ""),
                "reference": entry.get("reference") or entry.get("ground_truth", ""),
            }
        )
    return items


def _query_rag(pipeline, question: str, top_n: int = 3) -> dict:
    """Run a single query through the local RAG pipeline."""
    result = pipeline.run(question, top_n=top_n)
    return {
        "response": result["answer"],
        "retrieved_contexts": [doc["content"] for doc in result["sources"]],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline using Ragas 0.4.x"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval_dataset.json",
        help="Path to the evaluation dataset (JSON array)",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="data/eval_results.json",
        help="Path to save the detailed evaluation results",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default="data/eval_summary.json",
        help="Path to save the aggregated metric summary",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top reranked documents to pass to LLM",
    )
    # ---- Environment ----
    load_dotenv()
    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables.  "
            "Set it in .env or export it."
        )
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    parser.add_argument(
        "--gemini-model",
        type=str,
        default=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        help="Gemini model to use as the Ragas evaluator LLM",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate (useful for fast testing)",
    )
    args = parser.parse_args()

    # ---- Load dataset ----
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(
            f"Dataset file '{args.dataset}' not found. "
            "Create it manually or run 'make generate-testset'."
        )

    dataset_items = _load_dataset(args.dataset)
    if args.limit:
        dataset_items = dataset_items[:args.limit]
    print(f"Loaded {len(dataset_items)} evaluation questions from {args.dataset}")

    # ---- Initialise local RAG pipeline ----
    if not _HAS_RAG:
        raise ImportError(
            "Could not import MedicalRAGPipeline. "
            "Ensure src/application/pipeline_orchestrator.py is accessible."
        )
    print("Initialising local RAG pipeline …")
    rag_pipeline = MedicalRAGPipeline()

    # ---- Generate answers via RAG pipeline ----
    print("Querying RAG pipeline for each question …")
    samples: list[SingleTurnSample] = []

    for i, item in enumerate(dataset_items, 1):
        question = item["user_input"]
        reference = item["reference"]

        t0 = time.perf_counter()
        rag_out = _query_rag(rag_pipeline, question, top_n=args.top_n)
        elapsed = time.perf_counter() - t0

        sample = SingleTurnSample(
            user_input=question,
            response=rag_out["response"],
            retrieved_contexts=rag_out["retrieved_contexts"],
            reference=reference,
        )
        samples.append(sample)

        print(
            f"  [{i}/{len(dataset_items)}] {question[:60]}…  "
            f"({elapsed:.1f}s, {len(rag_out['retrieved_contexts'])} ctx)"
        )

    eval_dataset = EvaluationDataset(samples=samples)

    # ---- Ragas evaluator models ----
    print(f"\nInitialising Gemini evaluator ({args.gemini_model}) …")
    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=args.gemini_model)
    )
    # Prevent Gemini 400 error: 'Multiple candidates is not enabled for this model'
    evaluator_llm.bypass_n = True
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    from ragas.run_config import RunConfig

    # Limit concurrency to 2 workers to avoid hitting Gemini Free Tier rate limits (15 RPM)
    run_config = RunConfig(
        max_workers=int(os.getenv("RAGAS_MAX_WORKERS", "2")),
        timeout=180,
        max_retries=10,
        max_wait=60,
    )

    # ---- Run evaluation ----
    print("Running Ragas evaluation … (this calls Gemini API repeatedly)\n")
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )

    # ---- Results ----
    df = result.to_pandas()

    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    for col in metric_cols:
        if col in df.columns:
            valid = df[col].dropna()
            mean_val = valid.mean() if len(valid) > 0 else float("nan")
            coverage = len(valid)
            print(f"  {col:<25s}  mean={mean_val:.4f}  ({coverage}/{len(df)} scored)")
    print("=" * 60)

    # ---- Save detailed results ----
    df.to_json(args.out_file, orient="records", indent=2, force_ascii=False)
    print(f"\nDetailed results saved to {args.out_file}")

    # ---- Save summary ----
    summary = {}
    for col in metric_cols:
        if col in df.columns:
            valid = df[col].dropna()
            summary[col] = {
                "mean": round(float(valid.mean()), 4) if len(valid) > 0 else None,
                "std": round(float(valid.std()), 4) if len(valid) > 1 else None,
                "min": round(float(valid.min()), 4) if len(valid) > 0 else None,
                "max": round(float(valid.max()), 4) if len(valid) > 0 else None,
                "scored": int(len(valid)),
                "total": int(len(df)),
            }

    summary["_meta"] = {
        "dataset": args.dataset,
        "num_questions": len(dataset_items),
        "top_n": args.top_n,
        "evaluator_model": args.gemini_model,
    }

    with open(args.summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {args.summary_file}")


if __name__ == "__main__":
    main()