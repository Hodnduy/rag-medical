from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import QueryRequest, QueryResponse, SourceDocument
from src.api.dependencies import get_rag_pipeline
from src.application.pipeline_orchestrator import MedicalRAGPipeline

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok", "message": "Medical RAG API is running"}

@router.post("/api/chat", response_model=QueryResponse)
def chat_endpoint(
    request: QueryRequest, 
    pipeline: MedicalRAGPipeline = Depends(get_rag_pipeline)
):
    try:
        result = pipeline.run(user_query=request.query, top_n=request.top_n)
        
        sources = []
        for doc in result.get("sources", []):
            sources.append(SourceDocument(
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                dense_distance=doc.get("dense_distance")
            ))
            
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=sources
        )
    except Exception as e:
        print(f"[Error in /api/chat]: {e}")
        raise HTTPException(status_code=500, detail=str(e))
