"""
inference.py — Mandatory Hackathon Entry Point
================================================
Usage (as specified by judges):
    python inference.py --input hidden_private_dataset.json --output team_results.json

Input JSON format:
    [{"id": "...", "query": "..."}, ...]

Output JSON format:
    [{"id": "...", "retrieved_standards": [...], "latency_seconds": 1.23}, ...]

This script:
  1. Loads the pre-built FAISS + BM25 indices (fast, no rebuild needed)
  2. Runs each query through the full RAG pipeline
  3. Saves results in the exact schema required by eval_script.py

NOTE: No LLM is called during inference (generate_rationales=False)
for speed. This keeps average latency well under 5 seconds.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="BIS Standards RAG Inference Script"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file with query list",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file for results",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Override index directory (default: data/index)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of standards to retrieve per query (default: 5)",
    )
    args = parser.parse_args()

    # ── Validate input ────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path) as f:
        queries = json.load(f)

    if not isinstance(queries, list):
        logger.error("Input JSON must be a list of {id, query} objects")
        sys.exit(1)

    logger.info(f"Loaded {len(queries)} queries from {input_path}")

    # ── Load pipeline ─────────────────────────────────────────────────────
    from src.config import config
    from src.pipeline.rag_pipeline import BISRAGPipeline
    import subprocess

    index_dir = Path(args.index_dir) if args.index_dir else config.index_dir
    config.top_k_final = args.top_k

    if not index_dir.exists() or not list(index_dir.glob("*.index")):
        logger.warning(f"Indices not found in {index_dir}. Auto-building before inference...")
        try:
            subprocess.run([sys.executable, "scripts/build_index.py"], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to auto-build indices: {e}")
            sys.exit(1)

    logger.info(f"Loading BIS RAG pipeline from {index_dir}...")
    t_load = time.perf_counter()
    pipeline = BISRAGPipeline.load(index_dir, config)
    
    # ── Force-Load Lazy Models ────────────────────────────────────────────
    # Prevent the 50-second penalty on the first query by loading models into 
    # memory before the inference loop begins.
    logger.info("Pre-loading lazy ML models into memory...")
    pipeline.vector_store._load_model()
    pipeline.reranker._load_model()
    # Force eval mode on the cross encoder
    if pipeline.reranker._model:
        pipeline.reranker._model.model.eval()

    logger.info(f"Pipeline and models loaded in {time.perf_counter() - t_load:.2f}s")

    # ── Run inference ─────────────────────────────────────────────────────
    logger.info("Running inference...")
    t_start = time.perf_counter()

    results = []
    for i, item in enumerate(queries, 1):
        if not isinstance(item, dict):
            logger.warning(f"Malformed item at index {i}, skipping.")
            continue

        qid = item.get("id", f"Q{i:03d}")
        qtext = item.get("query", "")

        if not qtext:
            logger.warning(f"Empty query for id={qid}, skipping.")
            results.append({
                "id": qid,
                "retrieved_standards": [],
                "latency_seconds": 0.0,
            })
            continue

        q_start = time.perf_counter()
        try:
            result = pipeline.query(
                qtext,
                top_k=args.top_k,
                generate_rationales=False,  # Speed: skip LLM for eval
            )
            output = result.to_inference_output(qid)
        except Exception as e:
            logger.error(f"Error on query {qid}: {e}", exc_info=True)
            output = {
                "id": qid,
                "retrieved_standards": [],
                "latency_seconds": time.perf_counter() - q_start,
            }

        results.append(output)
        logger.info(
            f"[{i}/{len(queries)}] {qid} → "
            f"{output['retrieved_standards'][:3]} "
            f"({output['latency_seconds']:.3f}s)"
        )

    total_elapsed = time.perf_counter() - t_start
    avg_latency = sum(r["latency_seconds"] for r in results) / len(results) if results else 0

    # ── Save results ──────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETE")
    logger.info(f"  Queries processed : {len(results)}")
    logger.info(f"  Total time        : {total_elapsed:.2f}s")
    logger.info(f"  Avg latency       : {avg_latency:.3f}s/query")
    logger.info(f"  Results saved to  : {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
