"""
Cross-Encoder Reranker
=======================
Uses a cross-encoder (ms-marco-MiniLM-L-6-v2) to precisely score
(query, passage) pairs. Much more accurate than bi-encoder similarity
for final ranking but slower — applied only to the top-K candidates
from the hybrid retriever.

Cross-encoders jointly encode query + document, capturing fine-grained
interaction signals that bi-encoders miss (token-level attention).
"""
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks candidate (metadata, score) pairs using a cross-encoder.
    """

    def __init__(self, config):
        self.config = config
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder: {self.config.reranker_model}")
        self._model = CrossEncoder(
            self.config.reranker_model,
            max_length=512,
        )
        logger.info("Cross-encoder loaded.")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Dict[str, Any], float]],
        top_k: int = 5,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank candidates with the cross-encoder.

        Args:
            query: The user's query string
            candidates: List of (metadata, score) from hybrid retriever
            top_k: Number of final results to return

        Returns:
            Reranked list of (metadata, ce_score) sorted by ce_score desc
        """
        if not candidates:
            return []

        self._load_model()

        # Build (query, passage_text) pairs for the cross-encoder
        # Use title + category + is_number for the passage (concise but informative)
        pairs = []
        for meta, _ in candidates:
            passage = (
                f"{meta['is_number_full']} — {meta['title']}\n"
                f"Category: {meta.get('category', '')}\n"
                f"Section: {meta.get('section_name', '')}"
            ).strip()
            pairs.append([query, passage])

        # Score all pairs in one batched call
        scores = self._model.predict(
            pairs,
            batch_size=self.config.reranker_batch_size,
            show_progress_bar=False,
        )

        # Attach CE scores
        reranked = [
            (meta, float(ce_score))
            for (meta, _), ce_score in zip(candidates, scores)
        ]

        # Sort by CE score descending
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[:top_k]
