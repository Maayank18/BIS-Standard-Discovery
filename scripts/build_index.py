"""
build_index.py
==============
One-time script to:
  1. Parse the BIS SP 21 PDF
  2. Create multi-level chunks
  3. Build FAISS + BM25 indices
  4. Save to data/index/

Run: python scripts/build_index.py --pdf data/bis_sp21.pdf

After this completes, the API server and inference.py can load
the pre-built indices in seconds.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.ingestion.pdf_parser import BISPDFParser
from src.ingestion.chunker import create_chunks
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_index(pdf_path: Path, index_dir: Path, force: bool = False):
    """Full index build pipeline."""

    index_dir.mkdir(parents=True, exist_ok=True)

    # Check if index already exists
    if (index_dir / "faiss.index").exists() and not force:
        logger.info(
            f"Index already exists at {index_dir}. "
            "Use --force to rebuild."
        )
        return

    total_start = time.time()

    # ── Step 1: Parse PDF ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Parsing BIS SP 21 PDF")
    logger.info("=" * 60)
    parser = BISPDFParser(pdf_path)
    standards = parser.parse()

    if not standards:
        logger.error("No standards parsed from PDF! Check the PDF file.")
        sys.exit(1)

    logger.info(f"Parsed {len(standards)} BIS standards")

    # Print category distribution
    from collections import Counter
    cat_counts = Counter(s.category for s in standards if s.category)
    logger.info("Category distribution:")
    for cat, count in cat_counts.most_common():
        logger.info(f"  {cat}: {count} standards")

    # ── Step 2: Create Chunks ─────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Creating chunks")
    logger.info("=" * 60)
    chunks = create_chunks(standards)
    logger.info(f"Total chunks: {len(chunks)}")

    # ── Step 3: Build Vector Store (FAISS + BGE) ──────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Building FAISS vector index")
    logger.info("=" * 60)
    vector_store = VectorStore(config)
    vector_store.build(chunks)
    vector_store.save(index_dir)

    # ── Step 4: Build BM25 Index ──────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Building BM25 sparse index")
    logger.info("=" * 60)
    bm25 = BM25Retriever(config)
    bm25.build(chunks)
    bm25.save(index_dir)

    # ── Done ──────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info(f"Index build complete in {total_elapsed:.1f}s")
    logger.info(f"Index saved to: {index_dir}")
    logger.info(f"Standards indexed: {len(standards)}")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Build BIS RAG index from SP 21 PDF")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=config.pdf_path,
        help=f"Path to BIS SP 21 PDF (default: {config.pdf_path})",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=config.index_dir,
        help=f"Output directory for indices (default: {config.index_dir})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index exists",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        logger.error(
            f"PDF not found: {args.pdf}\n"
            "Download the BIS SP 21 PDF and place it at the specified path."
        )
        sys.exit(1)

    build_index(args.pdf, args.index_dir, force=args.force)


if __name__ == "__main__":
    main()
