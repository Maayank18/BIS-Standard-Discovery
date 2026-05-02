"""
Hybrid Retriever — Reciprocal Rank Fusion (RRF)
================================================
Combines dense (FAISS) and sparse (BM25) results using RRF.

RRF score for each document:  sum(1 / (k + rank_i))

Why RRF over weighted sum?
  - No normalisation needed across different score scales
  - Robust to outliers in either system
  - Empirically superior to simple weighted combination

After RRF fusion, we also apply:
  1. IS-number exact-match boost — if the query contains an explicit IS number
     that matches a retrieved standard, push it to rank 1.
  2. Category filter (optional) — if the query is clearly about a specific
     category (e.g., "cement"), boost standards from that category.
"""
import logging
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Matches IS numbers in queries: "IS 269", "IS 383:1970", "IS 2185 Part 2"
IS_IN_QUERY_PATTERN = re.compile(
    r"IS\s*(\d+(?:\s*\(Part\s*\d+\))?(?:\s*\(Sec\s*\d+\))?)"
    r"(?:\s*:?\s*(\d{4}))?",
    re.IGNORECASE,
)

# Keywords that suggest a specific category
CATEGORY_KEYWORDS = {
    "Cement and Concrete": [
        "cement", "concrete", "mortar", "grout", "aggregate", "clinker",
        "portland", "pozzolana", "slag", "fly ash", "opc", "ppc",
        "curing", "admixture", "blended",
    ],
    "Structural Steels": [
        "structural steel", "mild steel", "high yield", "deformed bar",
        "tmt", "tor", "angle section", "channel section",
    ],
    "Concrete Reinforcement": [
        "reinforcement", "rebar", "reinforcing bar", "stirrup", "wire mesh",
    ],
    "Stones": ["stone", "granite", "marble", "sandstone", "limestone quarry"],
    "Timber": ["timber", "hardwood", "softwood", "plywood", "particle board"],
    "Glass": ["glass", "glazing", "float glass"],
    "Bitumen and Tar Products": ["bitumen", "tar", "asphalt", "bituminous"],
    "Thermal Insulation Materials": ["insulation", "insulating", "thermal"],
}


def _detect_category(query: str) -> Optional[str]:
    """Return the most likely SP21 category from the query text."""
    query_lower = query.lower()
    best_category = None
    best_count = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in query_lower)
        if count > best_count:
            best_count = count
            best_category = category
    return best_category if best_count >= 1 else None


def _extract_is_numbers_from_query(query: str) -> List[str]:
    """
    Extract IS numbers explicitly mentioned in the query.
    Returns normalised list: ["is269", "is3831970"]
    """
    matches = IS_IN_QUERY_PATTERN.finditer(query)
    numbers = []
    for m in matches:
        num = m.group(1).strip().replace(" ", "").lower()
        year = m.group(2) or ""
        numbers.append(f"is{num}")
        if year:
            numbers.append(f"is{num}{year}")
    return numbers


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Dict[str, Any], float]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Fuse multiple ranked lists with RRF.

    Args:
        ranked_lists: List of [(metadata, score)] sorted by score desc
        k: RRF constant (60 is standard)
        weights: Optional per-list weights (default: equal)

    Returns:
        Fused list of (metadata, rrf_score) sorted by rrf_score desc
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    # Map chunk_id → (metadata, cumulative_rrf_score)
    scores: Dict[str, float] = defaultdict(float)
    chunk_meta: Dict[str, Dict] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (meta, _) in enumerate(ranked_list, start=1):
            chunk_id = meta["chunk_id"]
            scores[chunk_id] += weight * (1.0 / (k + rank))
            chunk_meta[chunk_id] = meta

    # Sort by RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [(chunk_meta[cid], scores[cid]) for cid in sorted_ids]


class HybridRetriever:
    """
    Hybrid retriever combining dense (FAISS) and sparse (BM25) retrieval
    via Reciprocal Rank Fusion with query-aware boosting.
    """

    def __init__(self, vector_store, bm25_retriever, config):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.config = config

    def retrieve(
        self,
        query: str,
        top_k: int = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Full hybrid retrieval pipeline.

        1. Dense retrieval (FAISS)
        2. Sparse retrieval (BM25)
        3. RRF fusion
        4. IS-number exact match boost
        5. Category boost (optional)

        Returns: list of (metadata, score) — deduplicated by IS number,
                 keeping the highest-scoring chunk per standard.
        """
        top_k = top_k or self.config.top_k_rerank

        # Step 1 & 2: Retrieve candidates from both systems
        dense_results = self.vector_store.search(
            query, top_k=self.config.top_k_dense
        )
        bm25_results = self.bm25_retriever.search(
            query, top_k=self.config.top_k_bm25
        )

        logger.debug(
            f"Dense: {len(dense_results)}, BM25: {len(bm25_results)} candidates"
        )

        # Step 3: RRF fusion — give slightly more weight to dense for
        # semantic queries but keep BM25 strong for IS-number queries
        fused = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            k=self.config.rrf_k,
            weights=[1.0, 1.2],  # BM25 slight boost for IS number precision
        )

        # Step 4: Deduplicate — one entry per IS standard (best chunk wins)
        deduped = self._dedup_by_standard(fused)

        # Step 5: IS-number exact match boost
        query_is_numbers = _extract_is_numbers_from_query(query)
        if query_is_numbers:
            deduped = self._boost_exact_is_match(deduped, query_is_numbers)

        # Step 6: Category boost
        detected_category = _detect_category(query)
        if detected_category:
            deduped = self._boost_category(deduped, detected_category, boost=0.15)

        return deduped[:top_k]

    def _dedup_by_standard(
        self, results: List[Tuple[Dict, float]]
    ) -> List[Tuple[Dict, float]]:
        """
        Keep only the highest-scoring chunk per IS standard.
        This ensures we return unique standards, not multiple chunks
        of the same standard.
        """
        seen: Dict[str, float] = {}
        best: Dict[str, Dict] = {}

        for meta, score in results:
            is_id = meta["is_number_full"]
            if is_id not in seen or score > seen[is_id]:
                seen[is_id] = score
                best[is_id] = meta

        # Sort by score
        return sorted(
            [(best[is_id], seen[is_id]) for is_id in seen],
            key=lambda x: x[1],
            reverse=True,
        )

    def _boost_exact_is_match(
        self,
        results: List[Tuple[Dict, float]],
        query_is_numbers: List[str],
    ) -> List[Tuple[Dict, float]]:
        """
        If query explicitly mentions an IS number, push that standard
        to the very top by giving it a large score bonus.
        """
        boosted = []
        for meta, score in results:
            is_norm = meta["is_number_full"].replace(" ", "").lower().replace(":", "")
            match = any(qn in is_norm for qn in query_is_numbers)
            new_score = score + (10.0 if match else 0.0)
            boosted.append((meta, new_score))

        return sorted(boosted, key=lambda x: x[1], reverse=True)

    def _boost_category(
        self,
        results: List[Tuple[Dict, float]],
        category: str,
        boost: float = 0.1,
    ) -> List[Tuple[Dict, float]]:
        """Apply a small score boost to standards in the detected category."""
        boosted = []
        for meta, score in results:
            if meta.get("category", "") == category:
                score += boost
            boosted.append((meta, score))
        return sorted(boosted, key=lambda x: x[1], reverse=True)
