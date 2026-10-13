"""
Generate evaluation testset using Ragas 0.4.x TestsetGenerator and Gemini API.

Reads cleaned NHS documents and uses Gemini to synthesise question-answer
pairs suitable for RAG evaluation.
"""

import json
import os
import sys
import types
import argparse
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

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_core.documents import Document


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation testset using Ragas and Gemini"
    )
    parser.add_argument(
        "--in-file",
        type=str,
        default="data/nhs_cleaned.jsonl",
        help="Input cleaned data (JSONL)",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="data/eval_testset.json",
        help="Output testset file (JSON)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=10,
        help="Number of test cases to generate",
    )
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
        default=os.getenv("GEMINI_MODEL", "gemini-3-flash"),
        help="Gemini model name for generation",
    )
    args = parser.parse_args()

    # ---- Load source documents ----
    print(f"Loading data from {args.in_file} …")
    documents: list[Document] = []
    with open(args.in_file, "r") as f:
        for line in f:
            data = json.loads(line)
            content = data.get("content", "")
            if content:
                metadata = {
                    "condition": data.get("condition", ""),
                    "section": data.get("section", ""),
                    "url": data.get("url", ""),
                }
                documents.append(Document(page_content=content, metadata=metadata))

    if not documents:
        raise ValueError("No documents loaded from the input file.")

    print(f"Loaded {len(documents)} documents.")

    # ---- Initialise Gemini wrappers ----
    print(f"Initialising Gemini ({args.gemini_model}) for testset generation …")
    generator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=args.gemini_model)
    )
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    )

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=embeddings,
    )

    # ---- Generate testset ----
    print(
        f"Generating testset of size {args.test_size} … (this may take a while)"
    )

    testset = generator.generate_with_langchain_docs(
        documents, testset_size=args.test_size
    )

    # ---- Save as JSON ----
    print(f"Saving testset to {args.out_file} …")
    testset_df = testset.to_pandas()

    # Normalise column names to our standard format
    rename_map = {}
    if "question" in testset_df.columns:
        rename_map["question"] = "user_input"
    if "ground_truth" in testset_df.columns:
        rename_map["ground_truth"] = "reference"
    if rename_map:
        testset_df = testset_df.rename(columns=rename_map)

    # Keep only the columns we need
    keep_cols = [c for c in ["user_input", "reference"] if c in testset_df.columns]
    testset_df[keep_cols].to_json(args.out_file, orient="records", indent=2)

    print(f"Done!  {len(testset_df)} test cases saved to {args.out_file}.")


if __name__ == "__main__":
    main()