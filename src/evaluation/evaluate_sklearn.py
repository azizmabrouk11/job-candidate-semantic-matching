"""
Simple sklearn-based evaluation.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def main():
    # Load data
    scores_path = Path("data/eval/pair_scores.json")
    with open(scores_path, 'r') as f:
        data = json.load(f)
    
    y_true = np.array([item['ground_truth_label'] for item in data])
    y_scores = np.array([item['semantic_score'] for item in data])
    
    print(f"Loaded {len(data)} pairs")
    print(f"Positive: {y_true.sum()}, Negative: {len(y_true) - y_true.sum()}\n")
    
    # ROC-AUC and Average Precision
    print("="*60)
    print(f"ROC-AUC:              {roc_auc_score(y_true, y_scores):.4f}")
    print(f"Average Precision:    {average_precision_score(y_true, y_scores):.4f}")
    print("="*60)
    
    # Evaluate at different thresholds
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*60)
    
    best_f1 = 0
    best_threshold = 0.65
    
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.68, 0.7, 0.75, 0.8]:
        y_pred = (y_scores >= threshold).astype(int)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1 = report['1']['f1-score']
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        
        marker = " ← BEST" if f1 == best_f1 else ""
        print(f"{threshold:<12.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}{marker}")
    
    # Detailed report for best threshold
    print(f"\n{'='*60}")
    print(f"Best Threshold: {best_threshold}")
    print("="*60)
    
    y_pred = (y_scores >= best_threshold).astype(int)
    print("\n" + classification_report(y_true, y_pred, target_names=['No Match', 'Match']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"              Predicted")
    print(f"           No Match  Match")
    print(f"Actual No  {cm[0][0]:7d}  {cm[0][1]:5d}")
    print(f"       Yes {cm[1][0]:7d}  {cm[1][1]:5d}")
    
    # Save results
    output = {
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "avg_precision": float(average_precision_score(y_true, y_scores)),
        "best_threshold": best_threshold,
        "best_f1": best_f1,
    }
    
    output_path = Path("data/eval/sklearn_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
