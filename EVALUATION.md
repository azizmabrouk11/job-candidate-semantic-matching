# Evaluation Guide

## Overview
You have 2502 labeled job-candidate pairs as ground truth. These labels were created using rule-based matching (experience + keyword overlap). Now you can evaluate how well your semantic matching system performs.

## Quick Start

### 1. Run Evaluation (Fast)
```bash
python src/evaluation/evaluate_fast.py
```

This computes semantic similarity scores for all pairs and compares them to ground truth at different thresholds.

### 2. View Results
```bash
python src/evaluation/visualize_results.py
```

This shows:
- Best threshold and metrics
- Performance comparison table
- Error analysis
- Recommendations

## What You'll Learn

### Key Metrics

**Precision**: Of all the matches you recommend, how many are actually good?
- High precision = Users trust your recommendations

**Recall**: Of all the good matches that exist, how many do you find?
- High recall = You don't miss opportunities

**F1-Score**: Harmonic mean of precision and recall
- Use this to pick the best threshold

### Example Output

```
EVALUATION RESULTS
======================================================================
Threshold    Precision    Recall       F1-Score     Accuracy    
----------------------------------------------------------------------
0.50         0.3245       0.8750       0.4732       0.6891      
0.55         0.4123       0.8125       0.5482       0.7234      
0.60         0.5234       0.7500       0.6154       0.7698      
0.65         0.6456       0.6875       0.6658       0.8123       ← BEST
0.70         0.7234       0.6250       0.6703       0.8345      
0.75         0.8123       0.5000       0.6188       0.8234      
```

In this example:
- **0.65** is the optimal threshold (highest F1)
- At 0.65: You correctly identify 66.58% of all pairs
- You have good balance (64.5% precision, 68.7% recall)

## Files Created

1. **data/eval/evaluation_results.json**
   - Summary metrics at all thresholds
   - Best threshold identification
   - Overall statistics

2. **data/eval/pair_scores.json**
   - Individual scores for each pair
   - Useful for error analysis
   - Can inspect specific false positives/negatives

## Interpreting Your Results

### Good Performance
- ✓ F1-Score > 0.70
- ✓ Precision and Recall both > 0.65
- ✓ Clear score separation between matches/non-matches

### Needs Work
- ⚠ F1-Score < 0.60
- ⚠ Very low precision (<0.5) or recall (<0.5)
- ⚠ No clear optimal threshold

### If Results Are Poor

1. **Check embeddings**: Are jobs and candidates properly embedded?
2. **Review search text**: Is `build_search_text.py` creating good descriptions?
3. **Verify ground truth**: Are rule-based labels reasonable?
4. **Examine errors**: Look at false positives/negatives in detail

## Advanced Analysis

### Custom Threshold
If you want to evaluate a specific threshold:

```python
from src.evaluation.evaluate_fast import main, evaluate_at_threshold
# Load data and run
# Then test custom threshold
metrics = evaluate_at_threshold(scores_with_labels, 0.68)
print(metrics)
```

### Error Investigation
Check `pair_scores.json` for problematic pairs:

```python
import json
with open("data/eval/pair_scores.json") as f:
    scores = json.load(f)

# Find high-scoring false positives
false_positives = [
    s for s in scores 
    if s["ground_truth_label"] == 0 and s["semantic_score"] > 0.8
]

# Investigate why these got high scores
for fp in false_positives[:5]:
    print(f"{fp['job_id']} + {fp['candidate_id']}: {fp['semantic_score']:.4f}")
```

## Next Steps

1. **Run evaluation now** to see baseline performance
2. **Note the optimal threshold** for use in production
3. **Analyze errors** to identify improvement opportunities
4. **Iterate**:
   - Improve embeddings (better search text)
   - Adjust rule-based ground truth if needed
   - Try different embedding models
   - Re-evaluate after changes

## Questions?

- Low precision? → Increase threshold or improve embedding quality
- Low recall? → Decrease threshold or enhance search text descriptions
- Both low? → Check if embeddings are working correctly
- High variance? → Need more training data or better features
