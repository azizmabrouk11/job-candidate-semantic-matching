"""
Standalone evaluation script that doesn't require pandas or MongoDB.
Works directly with Qdrant embeddings and ground truth JSON.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
from qdrant_client import QdrantClient

# Qdrant connection settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Calculate precision, recall, F1, and accuracy."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def get_all_embeddings(client: QdrantClient) -> Dict:
    """
    Retrieve all embeddings from Qdrant.
    
    Returns:
        Dictionary with job and candidate embeddings
    """
    print("Fetching all embeddings from Qdrant...")
    
    # Get all job embeddings
    jobs = {}
    try:
        job_records = client.scroll(
            collection_name="jobs",
            limit=1000,
            with_vectors=True,
        )[0]
        
        for record in job_records:
            mongodb_id = record.payload.get("mongodb_id")
            jobs[mongodb_id] = np.array(record.vector)
        
        print(f"  ✓ Loaded {len(jobs)} job embeddings")
    except Exception as e:
        print(f"  ✗ Error loading jobs: {e}")
        return None
    
    # Get all candidate embeddings
    candidates = {}
    try:
        candidate_records = client.scroll(
            collection_name="candidates",
            limit=1000,
            with_vectors=True,
        )[0]
        
        for record in candidate_records:
            mongodb_id = record.payload.get("mongodb_id")
            candidates[mongodb_id] = np.array(record.vector)
        
        print(f"  ✓ Loaded {len(candidates)} candidate embeddings")
    except Exception as e:
        print(f"  ✗ Error loading candidates: {e}")
        return None
    
    return {"jobs": jobs, "candidates": candidates}


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def compute_scores_fast(
    pairs: List[Dict], 
    embeddings: Dict
) -> List[tuple]:
    """
    Compute cosine similarity scores for all pairs using cached embeddings.
    
    Args:
        pairs: List of labeled pairs
        embeddings: Dictionary with 'jobs' and 'candidates' embeddings
    
    Returns:
        List of (job_id, candidate_id, label, score) tuples
    """
    results = []
    jobs_emb = embeddings["jobs"]
    candidates_emb = embeddings["candidates"]
    
    print(f"\nComputing similarity scores for {len(pairs)} pairs...")
    
    missing_jobs = set()
    missing_candidates = set()
    
    for i, pair in enumerate(pairs):
        job_id = pair["job_id"]
        candidate_id = pair["candidate_id"]
        label = pair["label_matched"]
        
        # Get embeddings
        job_vec = jobs_emb.get(job_id)
        cand_vec = candidates_emb.get(candidate_id)
        
        if job_vec is None:
            missing_jobs.add(job_id)
            score = 0.0
        elif cand_vec is None:
            missing_candidates.add(candidate_id)
            score = 0.0
        else:
            # Compute cosine similarity
            score = float(cosine_similarity(job_vec, cand_vec))
        
        results.append((job_id, candidate_id, label, score))
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} pairs...")
    
    if missing_jobs:
        print(f"\n⚠ Warning: {len(missing_jobs)} jobs not found in embeddings")
    if missing_candidates:
        print(f"⚠ Warning: {len(missing_candidates)} candidates not found in embeddings")
    
    return results


def evaluate_at_threshold(
    scores_with_labels: List[tuple], 
    threshold: float
) -> Dict:
    """Evaluate system at a specific threshold."""
    y_true = [label for _, _, label, _ in scores_with_labels]
    y_pred = [1 if score >= threshold else 0 for _, _, _, score in scores_with_labels]
    
    metrics = calculate_metrics(y_true, y_pred)
    metrics["threshold"] = threshold
    
    return metrics


def analyze_score_distribution(scores_with_labels: List[tuple]):
    """Analyze score distribution for matched vs non-matched pairs."""
    matched_scores = [score for _, _, label, score in scores_with_labels if label == 1]
    non_matched_scores = [score for _, _, label, score in scores_with_labels if label == 0]
    
    print("\n" + "="*70)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("="*70)
    
    if matched_scores:
        print(f"\nMatched pairs (label=1): n={len(matched_scores)}")
        print(f"  Mean:   {np.mean(matched_scores):.4f}")
        print(f"  Median: {np.median(matched_scores):.4f}")
        print(f"  Min:    {np.min(matched_scores):.4f}")
        print(f"  Max:    {np.max(matched_scores):.4f}")
        print(f"  Std:    {np.std(matched_scores):.4f}")
    
    if non_matched_scores:
        print(f"\nNon-matched pairs (label=0): n={len(non_matched_scores)}")
        print(f"  Mean:   {np.mean(non_matched_scores):.4f}")
        print(f"  Median: {np.median(non_matched_scores):.4f}")
        print(f"  Min:    {np.min(non_matched_scores):.4f}")
        print(f"  Max:    {np.max(non_matched_scores):.4f}")
        print(f"  Std:    {np.std(non_matched_scores):.4f}")


def main(ground_truth_path: str = "data/eval/pairs_labeled_rules.json"):
    """Main evaluation function."""
    print("="*70)
    print("SEMANTIC MATCHING SYSTEM EVALUATION")
    print("="*70)
    
    # Load ground truth
    print("\n1. Loading ground truth labels...")
    try:
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Ground truth file not found at {ground_truth_path}")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in {ground_truth_path}")
        return
    
    print(f"   ✓ Loaded {len(pairs)} labeled pairs")
    matched_count = sum(p["label_matched"] for p in pairs)
    print(f"     - Positive samples (matches): {matched_count}")
    print(f"     - Negative samples (no match): {len(pairs) - matched_count}")
    
    # Connect to Qdrant
    print("\n2. Connecting to Qdrant...")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections()
        print(f"   ✓ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as e:
        print(f"   ❌ Error connecting to Qdrant: {e}")
        print(f"   Make sure Qdrant is running on {QDRANT_HOST}:{QDRANT_PORT}")
        return
    
    # Get all embeddings at once (much faster)
    print("\n3. Fetching embeddings from Qdrant...")
    embeddings = get_all_embeddings(client)
    
    if embeddings is None:
        print("❌ Failed to load embeddings")
        return
    
    # Compute scores for all pairs
    print("\n4. Computing similarity scores...")
    scores_with_labels = compute_scores_fast(pairs, embeddings)
    
    # Analyze score distribution
    analyze_score_distribution(scores_with_labels)
    
    # Evaluate at different thresholds
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    best_f1 = 0
    best_threshold = 0
    
    results = []
    
    for threshold in thresholds:
        metrics = evaluate_at_threshold(scores_with_labels, threshold)
        results.append(metrics)
        
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_threshold = threshold
    
    # Print results table
    print("\n{:<12} {:<12} {:<12} {:<12} {:<12}".format(
        "Threshold", "Precision", "Recall", "F1-Score", "Accuracy"
    ))
    print("-" * 70)
    
    for metrics in results:
        marker = " ← BEST" if metrics["threshold"] == best_threshold else ""
        print("{:<12.2f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}{}".format(
            metrics["threshold"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["accuracy"],
            marker
        ))
    
    # Print best result details
    print("\n" + "="*70)
    print(f"🏆 BEST RESULT: Threshold = {best_threshold}")
    print("="*70)
    
    best_metrics = [m for m in results if m["threshold"] == best_threshold][0]
    print(f"\nPrecision:  {best_metrics['precision']:.4f}")
    print(f"Recall:     {best_metrics['recall']:.4f}")
    print(f"F1-Score:   {best_metrics['f1_score']:.4f}")
    print(f"Accuracy:   {best_metrics['accuracy']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  ┌─────────────┬─────────────┐")
    print(f"  │ TP: {best_metrics['tp']:7d} │ FP: {best_metrics['fp']:7d} │")
    print(f"  │ FN: {best_metrics['fn']:7d} │ TN: {best_metrics['tn']:7d} │")
    print(f"  └─────────────┴─────────────┘")
    
    # Save detailed results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    output_path = Path("data/eval/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save scores for further analysis
    scores_output = Path("data/eval/pair_scores.json")
    scores_data = [
        {
            "job_id": job_id,
            "candidate_id": cand_id,
            "ground_truth_label": label,
            "semantic_score": float(score),
        }
        for job_id, cand_id, label, score in scores_with_labels
    ]
    
    with open(scores_output, "w", encoding="utf-8") as f:
        json.dump(scores_data, f, indent=2)
    
    print(f"\n✓ Detailed scores saved to {scores_output}")
    
    # Save summary results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_threshold": best_threshold,
            "best_metrics": best_metrics,
            "all_results": results,
            "total_pairs": len(pairs),
            "positive_samples": matched_count,
            "negative_samples": len(pairs) - matched_count,
        }, f, indent=2)
    
    print(f"✓ Summary results saved to {output_path}")
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nRun: python src/evaluation/visualize_results.py")
    print("to see a detailed analysis of the results.")


if __name__ == "__main__":
    main()
