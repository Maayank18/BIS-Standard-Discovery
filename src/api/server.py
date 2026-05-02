"""
FastAPI Backend for BIS RAG Engine
====================================
Endpoints:
  POST /query          — single query, full rationale generation
  POST /batch          — batch query (for programmatic use)
  GET  /health         — health check + index stats
  GET  /standards      — list all indexed standards (for browse UI)
  GET  /categories     — list all SP21 categories
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import config
from src.pipeline.rag_pipeline import BISRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global pipeline instance (loaded once at startup)
pipeline: Optional[BISRAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline at startup."""
    global pipeline
    logger.info("Loading BIS RAG Pipeline...")
    try:
        pipeline = BISRAGPipeline.load(config.index_dir, config)
        logger.info("Pipeline loaded successfully.")
    except FileNotFoundError as e:
        logger.error(
            f"Index not found: {e}\n"
            "Run 'python scripts/build_index.py' to build the index first."
        )
    yield
    logger.info("Shutting down BIS RAG Engine.")


app = FastAPI(
    title="BIS Standards Recommendation Engine",
    description=(
        "AI-powered RAG system for recommending Bureau of Indian Standards "
        "(BIS SP 21) relevant to product descriptions. "
        "Built for the BIS × Sigma Squad Hackathon."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow React dev server and production build
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=2000,
                       description="Product description to find applicable BIS standards for")
    top_k: int = Field(5, ge=1, le=10, description="Number of standards to return")
    generate_rationales: bool = Field(True, description="Generate LLM rationales (slower but richer)")


class BatchQueryRequest(BaseModel):
    queries: List[Dict[str, str]] = Field(
        ..., description="List of {id, query} objects"
    )
    generate_rationales: bool = Field(False)


class StandardResult(BaseModel):
    is_number_full: str
    is_number: str
    year: str
    title: str
    category: str
    rationale: str
    score: float


class QueryResponse(BaseModel):
    query: str
    latency_seconds: float
    recommendations: List[StandardResult]
    total_found: int


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    total_standards: Optional[int]
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if pipeline is None:
        return HealthResponse(
            status="degraded",
            index_loaded=False,
            total_standards=None,
            message="Pipeline not loaded. Run build_index.py first.",
        )
    total = (
        pipeline.vector_store.index.ntotal
        if pipeline.vector_store.index
        else 0
    )
    return HealthResponse(
        status="healthy",
        index_loaded=True,
        total_standards=total,
        message=f"BIS RAG Engine running. {total} vectors indexed.",
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """
    Main query endpoint — returns top BIS standards for a product description.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not loaded. Please build the index first.",
        )
    if len(req.query.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Query too short. Provide a detailed product description.",
        )

    result = pipeline.query(
        req.query,
        top_k=req.top_k,
        generate_rationales=req.generate_rationales,
    )

    api_resp = result.to_api_response()
    return QueryResponse(
        query=api_resp["query"],
        latency_seconds=api_resp["latency_seconds"],
        recommendations=[StandardResult(**r) for r in api_resp["recommendations"]],
        total_found=len(api_resp["recommendations"]),
    )


@app.post("/batch")
async def batch_endpoint(req: BatchQueryRequest):
    """Batch query endpoint for programmatic evaluation."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded.")

    results = pipeline.batch_query(
        req.queries,
        generate_rationales=req.generate_rationales,
    )
    return {"results": results, "total": len(results)}


@app.get("/standards")
async def list_standards(
    category: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """List all indexed BIS standards (for browse UI)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded.")

    # Get all standard-level metadata from BM25 index
    all_meta = pipeline.bm25_retriever.metadata

    # Filter to unique standards only (standard-type chunks)
    seen = set()
    standards = []
    for meta in all_meta:
        if meta.get("chunk_type") != "standard":
            continue
        is_id = meta["is_number_full"]
        if is_id in seen:
            continue
        seen.add(is_id)

        # Apply category filter
        if category and meta.get("category", "").lower() != category.lower():
            continue

        # Apply search filter
        if search:
            search_lower = search.lower()
            if (
                search_lower not in meta.get("title", "").lower()
                and search_lower not in meta.get("is_number_full", "").lower()
            ):
                continue

        standards.append(
            {
                "is_number_full": meta["is_number_full"],
                "is_number": meta["is_number"],
                "year": meta["year"],
                "title": meta["title"],
                "category": meta.get("category", "General"),
                "page_start": meta.get("page_start", 0),
            }
        )

    # Sort by IS number
    standards.sort(key=lambda x: x["is_number_full"])
    total = len(standards)

    return {
        "standards": standards[offset: offset + limit],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@app.get("/categories")
async def list_categories():
    """List all SP21 categories with standard counts."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded.")

    all_meta = pipeline.bm25_retriever.metadata
    category_counts: Dict[str, int] = {}
    seen_standards = set()

    for meta in all_meta:
        if meta.get("chunk_type") != "standard":
            continue
        is_id = meta["is_number_full"]
        if is_id in seen_standards:
            continue
        seen_standards.add(is_id)
        cat = meta.get("category", "General")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    categories = [
        {"name": cat, "count": count}
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])
    ]
    return {"categories": categories}


# ── Example queries endpoint (for UI) ────────────────────────────────────────

@app.get("/examples")
async def get_examples():
    """Return example queries for the frontend."""
    return {
        "examples": [
            "We are a small enterprise manufacturing 33 Grade Ordinary Portland Cement. Which BIS standard covers the chemical and physical requirements?",
            "I need to comply with regulations for coarse and fine aggregates derived from natural sources for structural concrete.",
            "What is the official specification for precast concrete pipes for water mains?",
            "Our company manufactures hollow lightweight concrete masonry blocks. What standard applies?",
            "We produce Portland pozzolana cement using calcined clay. Which BIS standard governs our product?",
            "Looking for the standard for corrugated asbestos cement sheets for roofing and cladding.",
            "Which standard applies to White Portland cement for architectural and decorative purposes?",
            "We manufacture precast concrete coping blocks for wall tops. What standard should we follow?",
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host=config.api_host,
        port=config.api_port,
        reload=False,
        workers=1,
    )
