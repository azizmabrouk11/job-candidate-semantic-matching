"""
Visualize evaluation results and analyze system performance.
"""

import json
from pathlib import Path


def print_separator(char="=", length=70):
    """Print a separator line."""
    print(char * length)


def visualize_results(results_path: str = "data/eval/evaluation_results.json"):
    """
    Visualize evaluation results from JSON file.
    
    Args:
        results_path: Path to evaluation results JSON
    """
    # Load results
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print_separator()
    print("SEMANTIC MATCHING SYSTEM EVALUATION")
    print_separator()
    
    # Dataset info
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total pairs:      {data['total_pairs']}")
    print(f"  Positive samples: {data['positive_samples']} ({data['positive_samples']/data['total_pairs']*100:.1f}%)")
    print(f"  Negative samples: {data['negative_samples']} ({data['negative_samples']/data['total_pairs']*100:.1f}%)")
    
    # Best result
    best = data["best_metrics"]
    print_separator()
    print(f"🏆 BEST PERFORMANCE (Threshold = {data['best_threshold']})")
    print_separator()
    print(f"\n  Precision:  {best['precision']:.4f}")
    print(f"  Recall:     {best['recall']:.4f}")
    print(f"  F1-Score:   {best['f1_score']:.4f}")
    print(f"  Accuracy:   {best['accuracy']:.4f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"    ┌─────────────┬─────────────┐")
    print(f"    │ TP: {best['tp']:7d} │ FP: {best['fp']:7d} │")
    print(f"    │ FN: {best['fn']:7d} │ TN: {best['tn']:7d} │")
    print(f"    └─────────────┴─────────────┘")
    
    # All results comparison
    print("\n")
    print_separator()
    print("📈 PERFORMANCE AT DIFFERENT THRESHOLDS")
    print_separator()
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12}")
    print("-" * 70)
    
    for result in data["all_results"]:
        marker = " ★" if result["threshold"] == data["best_threshold"] else ""
        print(f"{result['threshold']:<12.2f} {result['precision']:<12.4f} {result['recall']:<12.4f} "
              f"{result['f1_score']:<12.4f} {result['accuracy']:<12.4f}{marker}")
    
    # Insights
    print("\n")
    print_separator()
    print("💡 INSIGHTS")
    print_separator()
    
    if best['precision'] > 0.8:
        print("\n✓ High precision: System rarely recommends bad matches")
    elif best['precision'] < 0.6:
        print("\n⚠ Low precision: Many false positives (bad matches recommended)")
    else:
        print("\n→ Moderate precision: Some false positives expected")
    
    if best['recall'] > 0.8:
        print("✓ High recall: System finds most good matches")
    elif best['recall'] < 0.6:
        print("⚠ Low recall: System misses many good matches")
    else:
        print("→ Moderate recall: Some good matches missed")
    
    if best['f1_score'] > 0.75:
        print("✓ Strong F1-score: Good balance between precision and recall")
    elif best['f1_score'] < 0.6:
        print("⚠ Low F1-score: Consider adjusting embeddings or rules")
    else:
        print("→ Moderate F1-score: Room for improvement")
    
    # Recommendations
    print("\n")
    print_separator()
    print("🎯 RECOMMENDATIONS")
    print_separator()
    
    print(f"\n1. Use threshold = {data['best_threshold']} for best F1-score")
    
    # Find high precision threshold
    high_precision = max(data["all_results"], key=lambda x: x["precision"])
    if high_precision["threshold"] != data["best_threshold"]:
        print(f"2. For higher precision (fewer false positives), use threshold = {high_precision['threshold']:.2f}")
        print(f"   → Precision: {high_precision['precision']:.4f}, Recall: {high_precision['recall']:.4f}")
    
    # Find high recall threshold
    high_recall = max(data["all_results"], key=lambda x: x["recall"])
    if high_recall["threshold"] != data["best_threshold"]:
        print(f"3. For higher recall (find more matches), use threshold = {high_recall['threshold']:.2f}")
        print(f"   → Precision: {high_recall['precision']:.4f}, Recall: {high_recall['recall']:.4f}")
    
    print("\n")


def analyze_pair_scores(scores_path: str = "data/eval/pair_scores.json"):
    """
    Analyze individual pair scores for error analysis.
    
    Args:
        scores_path: Path to pair scores JSON
    """
    if not Path(scores_path).exists():
        print(f"❌ Scores file not found: {scores_path}")
        return
    
    with open(scores_path, "r", encoding="utf-8") as f:
        scores = json.load(f)
    
    print_separator()
    print("🔍 ERROR ANALYSIS")
    print_separator()
    
    # Analyze false positives (predicted match, but ground truth says no)
    false_positives = [s for s in scores if s["ground_truth_label"] == 0 and s["semantic_score"] >= 0.7]
    print(f"\nFalse Positives at 0.7 threshold: {len(false_positives)}")
    
    if false_positives:
        print("\nTop 5 False Positives (High scores but not a match):")
        false_positives.sort(key=lambda x: x["semantic_score"], reverse=True)
        for i, fp in enumerate(false_positives[:5], 1):
            print(f"  {i}. {fp['job_id']} + {fp['candidate_id']}: score = {fp['semantic_score']:.4f}")
    
    # Analyze false negatives (predicted no match, but ground truth says yes)
    false_negatives = [s for s in scores if s["ground_truth_label"] == 1 and s["semantic_score"] < 0.7]
    print(f"\nFalse Negatives at 0.7 threshold: {len(false_negatives)}")
    
    if false_negatives:
        print("\nTop 5 False Negatives (Low scores but should match):")
        false_negatives.sort(key=lambda x: x["semantic_score"])
        for i, fn in enumerate(false_negatives[:5], 1):
            print(f"  {i}. {fn['job_id']} + {fn['candidate_id']}: score = {fn['semantic_score']:.4f}")
    
    print("\n")


def main():
    """Main visualization function."""
    results_path = "data/eval/evaluation_results.json"
    scores_path = "data/eval/pair_scores.json"
    
    if not Path(results_path).exists():
        print(f"❌ Results file not found: {results_path}")
        print("Run the evaluation first:")
        print("  python src/evaluation/evaluate_fast.py")
        return
    
    visualize_results(results_path)
    analyze_pair_scores(scores_path)


if __name__ == "__main__":
    main()
