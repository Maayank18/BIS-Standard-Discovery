"""
BM25 Sparse Retrieval Layer
============================
Uses rank_bm25 for keyword-based retrieval.
BM25 excels at exact IS number matches ("IS 269", "IS 383:1970")
and domain jargon ("supersulphated", "pozzolana") that embeddings
may diffuse into a generic region.

Serialised to disk as a pickle for fast reload.
"""
import json
import logging
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Stopwords lightweight list (domain-aware)
_STOPWORDS = frozenset([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "to", "for",
    "on", "at", "by", "from", "with", "and", "or", "but", "not", "no",
    "as", "if", "so", "it", "its", "this", "that", "these", "those",
    "we", "our", "their", "which", "what", "who", "where", "when", "how",
])


def _tokenize(text: str) -> List[str]:
    """
    Tokenise for BM25.
    Keeps IS-number tokens intact (e.g., "IS269", "is269:1989")
    and alphanumeric words.
    """
    # Lowercase
    text = text.lower()

    # Preserve IS numbers as tokens: "is 269 : 1989" → "is269" + "1989"
    text = re.sub(r"is\s*(\d+)", r"is\1", text)
    # Remove colons/punctuation except hyphens in compound words
    text = re.sub(r"[:\(\)]", " ", text)

    tokens = re.findall(r"[a-z0-9][a-z0-9\-]*", text)
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]
    return tokens


class BM25Retriever:
    """
    BM25-based sparse retriever.
    Indexes the same chunks as the vector store for hybrid fusion.
    """

    def __init__(self, config):
        self.config = config
        self.bm25 = None
        self.metadata: List[Dict[str, Any]] = []
        self._corpus_tokens: List[List[str]] = []

    # ── Build ─────────────────────────────────────────────────────────────

    def build(self, chunks) -> None:
        """Build BM25 index from chunks."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 not installed. Run: pip install rank-bm25")

        logger.info(f"Building BM25 index for {len(chunks)} chunks...")

        self.metadata = [c.to_metadata() for c in chunks]

        # Build corpus with enriched text (same as embedding text)
        corpus_texts = [c.get_embedding_text() for c in chunks]
        self._corpus_tokens = [_tokenize(t) for t in corpus_texts]

        self.bm25 = BM25Okapi(
            self._corpus_tokens,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        logger.info(f"BM25 index built: {len(self._corpus_tokens)} documents")

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        with open(index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "metadata": self.metadata,
                    "corpus_tokens": self._corpus_tokens,
                },
                f,
            )
        logger.info(f"BM25 index saved to {index_dir}")

    def load(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        pkl_path = index_dir / "bm25.pkl"

        if not pkl_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {index_dir}. Run build_index.py first."
            )

        logger.info(f"Loading BM25 index from {index_dir}...")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.metadata = data["metadata"]
        self._corpus_tokens = data["corpus_tokens"]
        logger.info(f"BM25 index loaded: {len(self.metadata)} documents")

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self, query: str, top_k: int = 30
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        BM25 retrieval for a query.
        Returns list of (metadata_dict, score) sorted by score descending.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 not loaded. Call load() or build() first.")

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append((self.metadata[idx], score))

        return results
