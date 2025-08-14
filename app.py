# app.py
import hashlib
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import Config
from engine import pinecone_index  # ✅ from engine/__init__.py
from engine.db import log_user_query
from engine.embedding_handler import embed_chunks
from engine.reasoner import generate_final_answer  # ✅ real function in your repo

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
    top_k: int = Field(5, description="Number of Pinecone matches to retrieve")

class QueryResponse(BaseModel):
    answer: str
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

    # Embed user query
    query_embedding = embed_chunks([request.query])[0]  # single query -> single embedding

    # Pinecone search
    if not pinecone_index:
        raise HTTPException(status_code=500, detail="Pinecone not initialized")

    search_results = pinecone_index.query(
        vector=query_embedding,
        top_k=request.top_k,
        include_metadata=True
    )

    # Prepare context
    sources = []
    context_texts = []
    for match in search_results.get("matches", []):
        chunk_text = match["metadata"].get("chunk", "")
        context_texts.append(chunk_text)
        sources.append({
            "id": match["id"],
            "score": match["score"],
            "chunk": chunk_text
        })

    # Generate final answer using RAG
    answer = generate_final_answer(request.query, "\n".join(context_texts))

    return QueryResponse(answer=answer, sources=sources)

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"})

