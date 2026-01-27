"""
Evaluate semantic matching system against rule-based ground truth.
Compares vector similarity scores with rule-based labels.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vector_store import QdrantStore
from src.db import get_jobs_df, get_candidates_df


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Calculate precision, recall, F1, and accuracy.
    
    Args:
        y_true: Ground truth labels (1=match, 0=no match)
        y_pred: Predicted labels (1=match, 0=no match)
    
    Returns:
        Dictionary with metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate metrics
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


def get_semantic_scores(
    pairs: List[Dict], 
    store: QdrantStore,
    jobs_dict: Dict,
    candidates_dict: Dict
) -> List[Tuple[str, str, int, float]]:
    """
    Get semantic similarity scores for each pair.
    
    Args:
        pairs: List of labeled pairs
        store: QdrantStore instance
        jobs_dict: Dictionary of job data by ID
        candidates_dict: Dictionary of candidate data by ID
    
    Returns:
        List of (job_id, candidate_id, label, score) tuples
    """
    results = []
    
    print(f"\nGetting semantic scores for {len(pairs)} pairs...")
    
    for i, pair in enumerate(pairs):
        job_id = pair["job_id"]
        candidate_id = pair["candidate_id"]
        label = pair["label_matched"]
        
        # Get candidate's search results for this job
        try:
            # Extract numeric ID from job_id (e.g., "job_001" -> 1)
            job_num = int(job_id.split("_")[1])
            cand_num = int(candidate_id.split("_")[1])
            
            # Search for this specific job-candidate pair
            # We'll search candidates for the job and see if this candidate is in results
            matches = store.search_candidates_for_job(job_num, limit=100)
            
            # Find the score for our specific candidate
            score = None
            for match in matches:
                if match["_id"] == candidate_id:
                    score = match["score"]
                    break
            
            # If not found in top 100, try reverse search
            if score is None:
                matches = store.search_jobs_for_candidate(cand_num, limit=100)
                for match in matches:
                    if match["_id"] == job_id:
                        score = match["score"]
                        break
            
            # If still not found, score is below threshold (set to 0)
            if score is None:
                score = 0.0
            
            results.append((job_id, candidate_id, label, score))
            
        except Exception as e:
            print(f"Error processing pair {job_id}-{candidate_id}: {e}")
            results.append((job_id, candidate_id, label, 0.0))
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(pairs)} pairs...")
    
    return results


def evaluate_at_threshold(
    scores_with_labels: List[Tuple[str, str, int, float]], 
    threshold: float
) -> Dict:
    """
    Evaluate system at a specific threshold.
    
    Args:
        scores_with_labels: List of (job_id, candidate_id, label, score)
        threshold: Similarity threshold (0.0 - 1.0)
    
    Returns:
        Metrics dictionary
    """
    y_true = [label for _, _, label, _ in scores_with_labels]
    y_pred = [1 if score >= threshold else 0 for _, _, _, score in scores_with_labels]
    
    metrics = calculate_metrics(y_true, y_pred)
    metrics["threshold"] = threshold
    
    return metrics


def main(ground_truth_path: str = "data/eval/pairs_labeled_rules.json"):
    """
    Main evaluation function.
    
    Args:
        ground_truth_path: Path to labeled pairs JSON
    """
    # Load ground truth
    print("Loading ground truth labels...")
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    print(f"Loaded {len(pairs)} labeled pairs")
    matched_count = sum(p["label_matched"] for p in pairs)
    print(f"  - Positive samples (matches): {matched_count}")
    print(f"  - Negative samples (no match): {len(pairs) - matched_count}")
    
    # Load data for lookups
    print("\nLoading job and candidate data...")
    jobs_df = get_jobs_df()
    candidates_df = get_candidates_df()
    jobs_dict = {str(row["_id"]): row for _, row in jobs_df.iterrows()}
    candidates_dict = {str(row["_id"]): row for _, row in candidates_df.iterrows()}
    
    # Initialize vector store
    print("\nConnecting to Qdrant...")
    store = QdrantStore()
    
    # Get semantic scores for all pairs
    scores_with_labels = get_semantic_scores(pairs, store, jobs_dict, candidates_dict)
    
    # Evaluate at different thresholds
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
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
        print("{:<12.2f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
            metrics["threshold"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["accuracy"],
        ))
    
    # Print best result details
    print("\n" + "="*70)
    print(f"BEST RESULT: Threshold = {best_threshold}")
    print("="*70)
    
    best_metrics = [m for m in results if m["threshold"] == best_threshold][0]
    print(f"Precision:  {best_metrics['precision']:.4f}")
    print(f"Recall:     {best_metrics['recall']:.4f}")
    print(f"F1-Score:   {best_metrics['f1_score']:.4f}")
    print(f"Accuracy:   {best_metrics['accuracy']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {best_metrics['tp']:4d}  FP: {best_metrics['fp']:4d}")
    print(f"  FN: {best_metrics['fn']:4d}  TN: {best_metrics['tn']:4d}")
    
    # Save results
    output_path = Path("data/eval/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_threshold": best_threshold,
            "best_metrics": best_metrics,
            "all_results": results,
            "total_pairs": len(pairs),
            "positive_samples": matched_count,
            "negative_samples": len(pairs) - matched_count,
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
