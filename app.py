# app.py
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import Config
from engine.db import log_user_query
from engine.reasoner import reason_over_query  # ✅ use new pipeline

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Banking Assistant API",
    description="RAG-first banking chatbot with Pinecone + Postgres logging",
    version="1.0.0"
)

# -----------------------------
# Request / Response Models
# -----------------------------
class QueryRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Unique user identifier")
    query: str = Field(..., description="User's question or statement")
    doc_id: str = Field(..., description="Document ID or namespace for retrieval")
    top_k: int = Field(5, description="Number of Pinecone matches to retrieve")

class QueryResponse(BaseModel):
    answer: dict
    sources: List[dict]

# -----------------------------
# API Key Verification
# -----------------------------
def verify_api_key(api_key: Optional[str]):
    if Config.API_KEY and api_key != Config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# -----------------------------
# Main Query Endpoint
# -----------------------------
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, x_api_key: Optional[str] = Header(None)):
    verify_api_key(x_api_key)

    # Log query to Postgres
    if request.user_id:
        log_user_query(request.user_id, request.query)

    # Call reasoner directly (handles parse → retrieve → LLM)
    result = reason_over_query(request.query, request.doc_id, top_k=request.top_k)

    # Extract sources from result["matched_clauses"]
    sources = [
        {
            "source": clause.get("source"),
            "doc_type": clause.get("doc_type"),
            "text": clause.get("text")
        }
        for clause in result.get("matched_clauses", [])
    ]

    return QueryResponse(answer=result, sources=sources)

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"})
