"""
FastAPI application for the SHL Assessment Recommender.

Exposes two endpoints:
- GET /health: Readiness check (returns {"status": "ok"})
- POST /chat: Stateless conversation endpoint
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import ChatRequest, ChatResponse
from app.agent import process_chat, get_catalog_store, get_retriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources at startup."""
    logger.info("Starting SHL Assessment Recommender...")
    start = time.time()
    
    # Pre-load catalog. The retriever is optional and lazy so the
    # deterministic recommender can serve hard-requirement checks quickly.
    catalog = get_catalog_store()
    logger.info(f"Catalog loaded: {len(catalog)} items")
    
    if os.environ.get("PRELOAD_RETRIEVER", "").strip().lower() in {"1", "true", "yes"}:
        retriever = get_retriever()
        logger.info(f"Retriever index loaded: {len(retriever.catalog)} items")
    
    elapsed = time.time() - start
    logger.info(f"Startup complete in {elapsed:.1f}s")
    
    yield
    
    logger.info("Shutting down SHL Assessment Recommender...")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL Individual Test Solutions",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for broad compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint.
    
    Returns {"status": "ok"} with HTTP 200.
    The evaluator allows up to 2 minutes for cold start.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return the agent's response.
    
    The API is stateless — every call carries the full conversation history.
    
    Request body:
        messages: List of {role: "user"|"assistant", content: str}
    
    Response:
        reply: Agent's natural language response
        recommendations: Empty [] or 1-10 assessment items
        end_of_conversation: true when task is complete
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")
    
    start_time = time.time()
    
    try:
        response = process_chat(request.messages)
        
        elapsed = time.time() - start_time
        logger.info(
            f"Chat processed in {elapsed:.1f}s | "
            f"Turns: {len(request.messages)} | "
            f"Recs: {len(response.recommendations)} | "
            f"EOC: {response.end_of_conversation}"
        )
        
        return response
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Chat failed after {elapsed:.1f}s: {e}", exc_info=True)
        
        # Return a graceful error response instead of 500
        return ChatResponse(
            reply="I encountered an issue processing your request. Could you please try rephrasing your question about SHL assessments?",
            recommendations=[],
            end_of_conversation=False,
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
