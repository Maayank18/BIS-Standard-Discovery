"""
validate_public.py
==================
Runs the pipeline against the public test set and reports metrics.
Validates that your system meets the hackathon targets BEFORE submission.

Usage:
    python scripts/validate_public.py

Saves results to data/public_results.json for eval_script.py.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PUBLIC_TEST_SET = [
    {"id": "PUB-01", "query": "We are a small enterprise manufacturing 33 Grade Ordinary Portland Cement. Which BIS standard covers the chemical and physical requirements for our product?", "expected_standards": ["IS 269: 1989"]},
    {"id": "PUB-02", "query": "I need to comply with the regulations for coarse and fine aggregates derived from natural sources intended for use in structural concrete.", "expected_standards": ["IS 383: 1970"]},
    {"id": "PUB-03", "query": "What is the official specification for manufacturing precast concrete pipes, both with and without reinforcement, for water mains?", "expected_standards": ["IS 458: 2003"]},
    {"id": "PUB-04", "query": "Our company is shifting to manufacturing hollow and solid lightweight concrete masonry blocks. What standard outlines the dimensions and physical requirements?", "expected_standards": ["IS 2185 (Part 2): 1983"]},
    {"id": "PUB-05", "query": "Looking for the standard detailing corrugated and semi-corrugated asbestos cement sheets used for roofing and cladding.", "expected_standards": ["IS 459: 1992"]},
    {"id": "PUB-06", "query": "What is the Indian Standard covering the manufacture, chemical, and physical requirements for Portland slag cement?", "expected_standards": ["IS 455: 1989"]},
    {"id": "PUB-07", "query": "We are setting up a plant to produce Portland pozzolana cement that is calcined clay based. What is the applicable standard?", "expected_standards": ["IS 1489 (Part 2): 1991"]},
    {"id": "PUB-08", "query": "Which standard applies to masonry cement used for general purposes where mortars for masonry are required, but not intended for structural concrete?", "expected_standards": ["IS 3466: 1988"]},
    {"id": "PUB-09", "query": "Looking for the standard that details the composition, manufacture, and testing of supersulphated cement, particularly for marine works or aggressive water conditions.", "expected_standards": ["IS 6909: 1990"]},
    {"id": "PUB-10", "query": "Our company manufactures White Portland cement for architectural and decorative purposes. Which standard governs its physical and chemical requirements?", "expected_standards": ["IS 8042: 1989"]},
]


def normalize_std(s: str) -> str:
    return s.replace(" ", "").lower()


def compute_metrics(results: list, ground_truth: list) -> dict:
    """Compute Hit Rate @3, MRR @5, Avg Latency."""
    gt_map = {item["id"]: item["expected_standards"] for item in ground_truth}
    hits_at_3 = 0
    mrr_sum = 0.0
    total_latency = 0.0
    n = len(results)

    per_query = []
    for res in results:
        qid = res["id"]
        expected = set(normalize_std(s) for s in gt_map.get(qid, []))
        retrieved = [normalize_std(s) for s in res.get("retrieved_standards", [])]
        latency = res.get("latency_seconds", 0.0)
        total_latency += latency

        # Hit@3
        hit = any(s in expected for s in retrieved[:3])
        if hit:
            hits_at_3 += 1

        # MRR@5
        mrr = 0.0
        for rank, s in enumerate(retrieved[:5], 1):
            if s in expected:
                mrr = 1.0 / rank
                break
        mrr_sum += mrr

        per_query.append({
            "id": qid,
            "hit@3": hit,
            "mrr@5": mrr,
            "latency": latency,
            "retrieved": retrieved[:5],
            "expected": list(expected),
        })

    return {
        "hit_rate_3": (hits_at_3 / n) * 100,
        "mrr_5": mrr_sum / n,
        "avg_latency": total_latency / n,
        "per_query": per_query,
    }


def main():
    from src.config import config
    from src.pipeline.rag_pipeline import BISRAGPipeline

    logger.info("Loading pipeline...")
    pipeline = BISRAGPipeline.load(config.index_dir, config)

    logger.info(f"Running {len(PUBLIC_TEST_SET)} public test queries...")
    results = pipeline.batch_query(PUBLIC_TEST_SET, generate_rationales=False)

    # Add expected standards to output for eval_script.py
    gt_map = {item["id"]: item["expected_standards"] for item in PUBLIC_TEST_SET}
    for r in results:
        r["expected_standards"] = gt_map.get(r["id"], [])

    # Save results
    output_path = Path("data/public_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Compute and display metrics
    metrics = compute_metrics(results, PUBLIC_TEST_SET)

    print("\n" + "=" * 60)
    print("  PUBLIC TEST SET VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Hit Rate @3  : {metrics['hit_rate_3']:.1f}%  (target >80%)")
    print(f"  MRR @5       : {metrics['mrr_5']:.4f}  (target >0.7)")
    print(f"  Avg Latency  : {metrics['avg_latency']:.3f}s  (target <5s)")
    print("=" * 60)

    print("\nPer-query breakdown:")
    for pq in metrics["per_query"]:
        status = "✓" if pq["hit@3"] else "✗"
        print(
            f"  {status} {pq['id']} | MRR={pq['mrr@5']:.2f} | "
            f"lat={pq['latency']:.3f}s | "
            f"retrieved={pq['retrieved'][:3]}"
        )

    print(f"\nFull results saved to: {output_path}")
    print("Run: python eval_script.py --results data/public_results.json")


if __name__ == "__main__":
    main()
