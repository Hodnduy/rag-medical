from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question: ")
    top_n: Optional[int] = Field(3, description="Number doc you want: ", ge=1, le=10)

class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    dense_distance: Optional[float] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceDocument]
