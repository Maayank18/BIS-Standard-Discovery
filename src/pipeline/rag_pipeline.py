"""
BIS RAG Pipeline — Main Orchestrator
======================================
End-to-end pipeline:

  Query
    │
    ▼
  HybridRetriever (FAISS + BM25 via RRF)
    │  top_k_rerank candidates (deduplicated by IS number)
    ▼
  CrossEncoderReranker
    │  top_k_final results
    ▼
  RationaleGenerator (LLM or template)
    │
    ▼
  RAGResult (is_numbers + rationales + metadata)

The pipeline is stateless after initialisation — safe for concurrent use.
"""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import Config, config as global_config
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.generation.rationale_generator import RationaleGenerator

logger = logging.getLogger(__name__)


@dataclass
class RecommendedStandard:
    """A single recommended BIS standard with rationale."""
    is_number_full: str      # e.g., "IS 269 : 1989"
    is_number: str           # e.g., "IS 269"
    year: str                # e.g., "1989"
    title: str               # standard title
    category: str            # SP21 category
    rationale: str           # why this standard applies
    retrieval_score: float   # final reranker score


@dataclass
class RAGResult:
    """Complete result from a single RAG query."""
    query: str
    recommendations: List[RecommendedStandard]
    latency_seconds: float
    retrieved_standards: List[str]   # ordered IS number strings (for eval)

    def to_inference_output(self, query_id: str) -> Dict:
        """Format for inference.py output JSON (hackathon schema)."""
        return {
            "id": query_id,
            "retrieved_standards": self.retrieved_standards,
            "latency_seconds": round(self.latency_seconds, 4),
        }

    def to_api_response(self) -> Dict:
        """Rich format for the API / frontend."""
        return {
            "query": self.query,
            "latency_seconds": round(self.latency_seconds, 4),
            "recommendations": [
                {
                    "is_number_full": r.is_number_full,
                    "is_number": r.is_number,
                    "year": r.year,
                    "title": r.title,
                    "category": r.category,
                    "rationale": r.rationale,
                    "score": round(r.retrieval_score, 4),
                }
                for r in self.recommendations
            ],
        }


class BISRAGPipeline:
    """
    Fully assembled BIS RAG pipeline. Call .query() to get recommendations.
    Initialise with BISRAGPipeline.load() to load persisted indices.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        reranker: CrossEncoderReranker,
        rationale_generator: RationaleGenerator,
        config: Config,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.hybrid_retriever = HybridRetriever(
            vector_store, bm25_retriever, config
        )
        self.reranker = reranker
        self.rationale_generator = rationale_generator
        self.config = config

    @classmethod
    def load(
        cls,
        index_dir: Optional[Path] = None,
        config: Optional[Config] = None,
    ) -> "BISRAGPipeline":
        """
        Factory method: load all components from persisted index.
        This is what inference.py and the API use.
        """
        cfg = config or global_config
        idx_dir = Path(index_dir) if index_dir else cfg.index_dir

        vector_store = VectorStore(cfg)
        vector_store.load(idx_dir)

        bm25_retriever = BM25Retriever(cfg)
        bm25_retriever.load(idx_dir)

        reranker = CrossEncoderReranker(cfg)
        rationale_generator = RationaleGenerator(cfg)

        pipeline = cls(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            reranker=reranker,
            rationale_generator=rationale_generator,
            config=cfg,
        )
        logger.info("BIS RAG Pipeline loaded and ready.")
        return pipeline

    def query(
        self,
        query_text: str,
        top_k: int = None,
        generate_rationales: bool = True,
    ) -> RAGResult:
        """
        Run the full RAG pipeline for a single query.

        Args:
            query_text: Product description from user/eval script
            top_k: Override default number of results
            generate_rationales: If False, skip LLM call (faster for eval)

        Returns:
            RAGResult with ordered recommendations
        """
        t_start = time.perf_counter()
        top_k = top_k or self.config.top_k_final

        # ── Stage 1: Hybrid Retrieval ─────────────────────────────────────
        candidates = self.hybrid_retriever.retrieve(
            query_text,
            top_k=self.config.top_k_rerank,
        )
        logger.debug(f"Hybrid retrieval returned {len(candidates)} candidates")

        # ── Stage 2: Cross-Encoder Reranking ──────────────────────────────
        import torch
        with torch.no_grad():
            reranked = self.reranker.rerank(query_text, candidates, top_k=top_k)
        logger.debug(f"Reranker returned {len(reranked)} results")

        # ── Stage 3: Rationale Generation ────────────────────────────────
        reranked_metas = [meta for meta, _ in reranked]
        reranked_scores = [score for _, score in reranked]

        if generate_rationales:
            rationale_items = self.rationale_generator.generate(
                query_text, reranked_metas
            )
            # ── Assemble result ───────────────────────────────────────────────
            recommendations = []
            for item, score in zip(rationale_items, reranked_scores):
                recommendations.append(
                    RecommendedStandard(
                        is_number_full=item["is_number_full"],
                        is_number=item.get("is_number", ""),
                        year=item.get("year", ""),
                        title=item.get("title", ""),
                        category=item.get("category", ""),
                        rationale=item.get("rationale", ""),
                        retrieval_score=score,
                    )
                )

            latency = time.perf_counter() - t_start
            result = RAGResult(
                query=query_text,
                recommendations=recommendations,
                latency_seconds=latency,
                retrieved_standards=[r.is_number_full for r in recommendations],
            )
        else:
            # ── Hardcoded LLM Bypass for Automated Eval ───────────────────
            latency = time.perf_counter() - t_start
            result = RAGResult(
                query=query_text,
                recommendations=[],  # Skipped entirely
                latency_seconds=latency,
                retrieved_standards=[m["is_number_full"] for m in reranked_metas[:top_k]],
            )

        logger.info(
            f"Query completed in {latency:.3f}s | "
            f"Top result: {result.retrieved_standards[0] if result.retrieved_standards else 'none'}"
        )
        return result

    def batch_query(
        self,
        queries: List[Dict],  # list of {"id": ..., "query": ...}
        generate_rationales: bool = False,
    ) -> List[Dict]:
        """
        Batch inference — used by inference.py.
        generate_rationales=False by default for speed (eval only needs IS numbers).
        """
        results = []
        for item in queries:
            qid = item.get("id", "")
            qtext = item.get("query", "")
            try:
                result = self.query(qtext, generate_rationales=generate_rationales)
                results.append(result.to_inference_output(qid))
            except Exception as e:
                logger.error(f"Error processing query {qid}: {e}", exc_info=True)
                results.append(
                    {
                        "id": qid,
                        "query": qtext,
                        "retrieved_standards": [],
                        "latency_seconds": 0.0,
                    }
                )
        return results
