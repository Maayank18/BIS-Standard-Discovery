"""
Vector Store — FAISS + BGE Embeddings
======================================
Dense retrieval layer using:
  - BAAI/bge-base-en-v1.5 embeddings (768-dim, strong on technical text)
  - FAISS IndexFlatIP (inner product = cosine similarity on L2-normalised vectors)
  - Metadata stored separately in a JSON sidecar

Build once with build_index(), then load with load_index() for fast inference.
"""
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-backed dense vector store with BGE embeddings.
    Thread-safe for concurrent reads (FAISS is GIL-friendly for searches).
    """

    def __init__(self, config):
        self.config = config
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_texts: List[str] = []
        self._model = None
        self._tokenizer = None

    # ── Embedding ─────────────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self._model = SentenceTransformer(self.config.embedding_model)
        logger.info("Embedding model loaded.")

    def _embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Embed a list of texts.
        BGE models require a query instruction prefix for queries only.
        """
        self._load_model()
        if is_query:
            texts = [self.config.query_instruction + t for t in texts]

        embeddings = self._model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,   # L2 normalize → cosine via inner product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    # ── Index Build ───────────────────────────────────────────────────────

    def build(self, chunks) -> None:
        """
        Build FAISS index from chunks.
        chunks: List[Chunk] from chunker.py
        """
        import faiss

        logger.info(f"Building FAISS index for {len(chunks)} chunks...")
        t0 = time.time()

        # Prepare texts and metadata
        embed_texts = [c.get_embedding_text() for c in chunks]
        self.metadata = [c.to_metadata() for c in chunks]
        self.embedding_texts = embed_texts

        # Embed in batches
        embeddings = self._embed(embed_texts, is_query=False)

        # Build FAISS index
        dim = embeddings.shape[1]
        # FlatIP on normalised vectors = cosine similarity
        self.index = faiss.IndexFlatIP(dim)

        # For large datasets, use IVF for speed
        if len(chunks) > 10_000:
            nlist = min(int(len(chunks) ** 0.5), 512)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.nprobe = min(nlist // 4, 64)

        self.index.add(embeddings)

        elapsed = time.time() - t0
        logger.info(
            f"FAISS index built: {self.index.ntotal} vectors, "
            f"dim={dim}, time={elapsed:.1f}s"
        )

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, index_dir: Path) -> None:
        """Save FAISS index + metadata to disk."""
        import faiss

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_dir / "faiss.index"))

        with open(index_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Vector store saved to {index_dir}")

    def load(self, index_dir: Path) -> None:
        """Load FAISS index + metadata from disk."""
        import faiss

        index_dir = Path(index_dir)
        index_path = index_dir / "faiss.index"
        meta_path = index_dir / "metadata.json"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Index not found at {index_dir}. Run build_index.py first."
            )

        logger.info(f"Loading FAISS index from {index_dir}...")
        self.index = faiss.read_index(str(index_path))

        with open(meta_path) as f:
            self.metadata = json.load(f)

        logger.info(
            f"Vector store loaded: {self.index.ntotal} vectors, "
            f"{len(self.metadata)} metadata records"
        )

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self, query: str, top_k: int = 30
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Dense retrieval for a query.
        Returns list of (metadata_dict, score) sorted by score descending.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() or build() first.")

        query_vec = self._embed([query], is_query=True)  # shape (1, dim)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))

        return results
