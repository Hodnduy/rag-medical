from functools import lru_cache
from src.application.pipeline_orchestrator import MedicalRAGPipeline

@lru_cache(maxsize=1)
def get_rag_pipeline() -> MedicalRAGPipeline:
    print("[Info] Initilize MedicalRAGPipeline...")
    pipeline = MedicalRAGPipeline(top_k_dense=15, top_k_sparse=15)
    return pipeline
