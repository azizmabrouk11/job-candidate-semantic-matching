<div align="center">

# 📊 Evaluation Guide

### *Comprehensive Testing & Performance Analysis Framework*

---

</div>

## 🎯 Overview

Your system includes **2500+ labeled job-candidate pairs** as ground truth, created using intelligent rule-based matching (experience validation + keyword overlap). This evaluation framework helps you measure how well your semantic matching system performs against these labels.

### 🔬 What You'll Measure

- ✅ **Precision** - Trust in your recommendations
- ✅ **Recall** - Coverage of good matches
- ✅ **F1-Score** - Balance between precision and recall
- ✅ **ROC-AUC** - Overall discriminative ability
- ✅ **Optimal Threshold** - Best operating point


---

## ⚡ Quick Start

### Method 1: Fast Batch Evaluation

```bash
python src/evaluation/evaluate_fast.py
```

**What it does:**
- 📊 Computes semantic similarity scores for all labeled pairs
- 🎯 Tests multiple thresholds (0.5 - 0.9)
- 🏆 Identifies optimal threshold by F1-score
- 💾 Saves results to `data/eval/evaluation_results.json`

### Method 2: sklearn-Based Evaluation (NEW! ⭐)

```bash
python src/evaluation/evaluate_sklearn.py
```

**What it does:**
- 🤖 Professional metrics using scikit-learn
- 📈 ROC-AUC and Average Precision scores
- 📋 Detailed classification reports
- 🎨 Confusion matrix analysis
- 💾 Saves to `data/eval/sklearn_results.json`

### Method 3: Visualize Results

```bash
python src/evaluation/visualize_results.py
```

**What it shows:**
- 📊 Best threshold and performance metrics
- 📈 Performance comparison across thresholds
- ❌ Error analysis (false positives/negatives)
- 💡 Recommendations for improvement

---


## 📚 Understanding Key Metrics

### 🎯 Precision
**Formula:** `TP / (TP + FP)`

**What it means:** Of all the matches you recommend, how many are actually good?

- ✅ **High precision** = Users trust your recommendations
- ⚠️ **Low precision** = Too many false alarms

**Example:** If you recommend 100 matches and 80 are correct → **80% precision**

---

### 🔍 Recall  
**Formula:** `TP / (TP + FN)`

**What it means:** Of all the good matches that exist, how many do you find?

- ✅ **High recall** = You don't miss opportunities
- ⚠️ **Low recall** = Missing good candidates

**Example:** If there are 100 good matches and you find 70 → **70% recall**

---

### ⚖️ F1-Score
**Formula:** `2 × (Precision × Recall) / (Precision + Recall)`

**What it means:** Harmonic mean balancing precision and recall

- ✅ **Use this to pick the best threshold**
- 🎯 Higher F1 = Better balance

---

### 📈 ROC-AUC
**Formula:** Area under the Receiver Operating Characteristic curve

**What it means:** Overall ability to distinguish matches from non-matches

| Score Range | Quality |
|-------------|---------|
| 0.90 - 1.00 | 🏆 Excellent |
| 0.80 - 0.90 | ✨ Very Good |
| 0.70 - 0.80 | ⚡ Good |
| 0.60 - 0.70 | 📌 Fair |
| < 0.60 | ⚠️ Needs Work |

---

### 🎨 Average Precision
**Formula:** Area under the Precision-Recall curve

**What it means:** Average precision across all recall levels

- Better for imbalanced datasets (like job matching)
- Focus on positive class performance

---


## 📊 Example Output

### Fast Evaluation Output

```
════════════════════════════════════════════════════════════════════════
                        EVALUATION RESULTS
════════════════════════════════════════════════════════════════════════
Loaded 2502 labeled pairs
  - Positive samples (matches): 834
  - Negative samples (no match): 1668

────────────────────────────────────────────────────────────────────────
Threshold    Precision    Recall       F1-Score     Accuracy    
────────────────────────────────────────────────────────────────────────
0.50         0.3245       0.8750       0.4732       0.6891      
0.55         0.4123       0.8125       0.5482       0.7234      
0.60         0.5234       0.7500       0.6154       0.7698      
0.65         0.6456       0.6875       0.6658       0.8123      ← BEST
0.70         0.7234       0.6250       0.6703       0.8345      
0.75         0.8123       0.5000       0.6188       0.8234      
0.80         0.8756       0.3750       0.5245       0.7891      

════════════════════════════════════════════════════════════════════════
BEST RESULT: Threshold = 0.65
════════════════════════════════════════════════════════════════════════
Precision:  0.6456
Recall:     0.6875
F1-Score:   0.6658
Accuracy:   0.8123

Confusion Matrix:
  TP:  574    FP:  315
  FN:  260    TN: 1353
════════════════════════════════════════════════════════════════════════
```

**Interpretation:**
- 🎯 **Optimal threshold**: 0.65 (highest F1-score)
- ✅ **Good balance**: 64.56% precision, 68.75% recall
- 💪 **Strong accuracy**: 81.23% overall correctness

---

### sklearn Evaluation Output

```
════════════════════════════════════════════════════════════════════════
                    SKLEARN-BASED EVALUATION
════════════════════════════════════════════════════════════════════════
Loaded 2502 pairs
Positive: 834, Negative: 1668

════════════════════════════════════════════════════════════════════════
ROC-AUC:              0.8456
Average Precision:    0.8123
════════════════════════════════════════════════════════════════════════

Threshold    Precision    Recall       F1          
────────────────────────────────────────────────────
0.50         0.3245       0.8750       0.4732      
0.55         0.4123       0.8125       0.5482      
0.60         0.5234       0.7500       0.6154      
0.65         0.6456       0.6875       0.6658      ← BEST
0.68         0.6789       0.6500       0.6641      
0.70         0.7234       0.6250       0.6703      
0.75         0.8123       0.5000       0.6188      
0.80         0.8756       0.3750       0.5245      

════════════════════════════════════════════════════════════════════════
Best Threshold: 0.65
════════════════════════════════════════════════════════════════════════

              precision    recall  f1-score   support

    No Match     0.8123    0.8456    0.8286      1668
       Match     0.6456    0.6875    0.6658       834

    accuracy                         0.7845      2502

Confusion Matrix:
              Predicted
           No Match  Match
Actual No      1410    258
       Yes      261    573

✓ Results saved to data/eval/sklearn_results.json
```

---


## 📁 Output Files

### 1. `data/eval/evaluation_results.json`
**Generated by:** `evaluate_fast.py`

```json
{
  "best_threshold": 0.65,
  "best_metrics": {
    "precision": 0.6456,
    "recall": 0.6875,
    "f1_score": 0.6658,
    "accuracy": 0.8123,
    "tp": 574,
    "fp": 315,
    "tn": 1353,
    "fn": 260
  },
  "all_results": [...],
  "total_pairs": 2502,
  "positive_samples": 834,
  "negative_samples": 1668
}
```

**Use for:**
- Quick performance summary
- Threshold comparison
- Overall system statistics

---

### 2. `data/eval/pair_scores.json`
**Generated by:** `evaluate_fast.py`

```json
[
  {
    "job_id": "job_001",
    "candidate_id": "cand_042",
    "ground_truth_label": 1,
    "semantic_score": 0.7234
  },
  ...
]
```

**Use for:**
- Error analysis (false positives/negatives)
- Individual pair inspection
- Score distribution analysis
- Debugging specific matches

---

### 3. `data/eval/sklearn_results.json`
**Generated by:** `evaluate_sklearn.py` ⭐

```json
{
  "roc_auc": 0.8456,
  "avg_precision": 0.8123,
  "best_threshold": 0.65,
  "best_f1": 0.6658
}
```

**Use for:**
- Professional reporting
- Academic/research documentation
- Threshold-independent metrics
- Model comparison

---


## 🎯 Interpreting Your Results

### 🏆 Excellent Performance

✅ **Indicators:**
- F1-Score **> 0.70**
- Precision and Recall **both > 0.65**
- ROC-AUC **> 0.85**
- Clear score separation between matches/non-matches

💪 **What this means:**
- Your semantic matching is highly accurate
- Ready for production deployment
- Users will trust recommendations

---

### ⚡ Good Performance

✅ **Indicators:**
- F1-Score **0.60 - 0.70**
- ROC-AUC **0.75 - 0.85**
- Reasonable precision/recall balance

💡 **What to do:**
- Fine-tune threshold for your use case
- Consider A/B testing
- Monitor in production

---

### ⚠️ Needs Improvement

❌ **Warning signs:**
- F1-Score **< 0.60**
- Very low precision (**< 0.5**) or recall (**< 0.5**)
- ROC-AUC **< 0.70**
- No clear optimal threshold

🔧 **Action items:**
1. **Check embeddings** - Are jobs and candidates properly embedded?
2. **Review search text** - Is `build_search_text.py` creating good descriptions?
3. **Verify ground truth** - Are rule-based labels reasonable?
4. **Examine errors** - Look at false positives/negatives in detail

---

## 🔍 Troubleshooting Guide

### Problem: Low Precision (Too Many False Positives)

**Symptoms:**
- High semantic scores for non-matching pairs
- Users complain about irrelevant recommendations

**Solutions:**
```bash
# 1. Increase threshold
python scripts/match_candidate.py 19 0.75  # Instead of 0.65

# 2. Improve search text quality
# Edit src/db/data_loader.py to include more relevant fields

# 3. Re-evaluate
python src/evaluation/evaluate_sklearn.py
```

---

### Problem: Low Recall (Missing Good Matches)

**Symptoms:**
- Good matches have low semantic scores
- Users find matches you missed

**Solutions:**
```bash
# 1. Decrease threshold
python scripts/match_candidate.py 19 0.55  # Instead of 0.65

# 2. Enhance embeddings
# Make sure search_text includes all relevant information
python scripts/build_search_text.py

# Re-embed with improved search text
python scripts/embed_jobs.py
python scripts/embed_candidates.py

# 3. Re-evaluate
python src/evaluation/evaluate_fast.py
```

---

### Problem: Both Low (System Issues)

**Symptoms:**
- Both precision and recall are poor
- ROC-AUC < 0.60

**Solutions:**

1. **Verify embeddings exist:**
```bash
python -c "from src.vector_store import QdrantStore; print(QdrantStore().get_collections())"
```

2. **Check embedding quality:**
```bash
# Re-embed everything
python scripts/embed_jobs.py
python scripts/embed_candidates.py
```

3. **Review ground truth labels:**
```python
import json
with open("data/eval/pairs_labeled_rules.json") as f:
    pairs = json.load(f)

# Check label distribution
matches = sum(p["label_matched"] for p in pairs)
print(f"Matches: {matches}/{len(pairs)} ({matches/len(pairs)*100:.1f}%)")
```

4. **Test with single example:**
```bash
python scripts/match_candidate.py 19 0.5
```

---


## 🔬 Advanced Analysis

### Custom Threshold Testing

Test specific thresholds programmatically:

```python
from src.evaluation.evaluate_fast import evaluate_at_threshold
import json

# Load scored pairs
with open("data/eval/pair_scores.json") as f:
    scores_data = json.load(f)

# Convert to expected format
scores_with_labels = [
    (item['job_id'], item['candidate_id'], 
     item['ground_truth_label'], item['semantic_score'])
    for item in scores_data
]

# Test custom threshold
metrics = evaluate_at_threshold(scores_with_labels, 0.68)
print(f"F1-Score at 0.68: {metrics['f1_score']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
```

---

### Error Investigation

Identify and analyze problematic matches:

```python
import json

with open("data/eval/pair_scores.json") as f:
    scores = json.load(f)

# Find high-scoring false positives
false_positives = [
    s for s in scores 
    if s["ground_truth_label"] == 0 and s["semantic_score"] > 0.75
]

print(f"Found {len(false_positives)} high-scoring false positives")
for fp in false_positives[:5]:
    print(f"Job {fp['job_id']} + Candidate {fp['candidate_id']}: "
          f"Score = {fp['semantic_score']:.4f}")

# Find low-scoring true positives (missed matches)
false_negatives = [
    s for s in scores 
    if s["ground_truth_label"] == 1 and s["semantic_score"] < 0.60
]

print(f"\nFound {len(false_negatives)} low-scoring true matches")
for fn in false_negatives[:5]:
    print(f"Job {fn['job_id']} + Candidate {fn['candidate_id']}: "
          f"Score = {fn['semantic_score']:.4f}")
```

---

### Score Distribution Analysis

Understand how scores are distributed:

```python
import json
import numpy as np

with open("data/eval/pair_scores.json") as f:
    scores = json.load(f)

# Separate by label
match_scores = [s["semantic_score"] for s in scores if s["ground_truth_label"] == 1]
no_match_scores = [s["semantic_score"] for s in scores if s["ground_truth_label"] == 0]

print("Match Scores (should be HIGH):")
print(f"  Mean: {np.mean(match_scores):.4f}")
print(f"  Median: {np.median(match_scores):.4f}")
print(f"  Std: {np.std(match_scores):.4f}")

print("\nNo-Match Scores (should be LOW):")
print(f"  Mean: {np.mean(no_match_scores):.4f}")
print(f"  Median: {np.median(no_match_scores):.4f}")
print(f"  Std: {np.std(no_match_scores):.4f}")

# Check for overlap
overlap = len([s for s in match_scores if s < np.mean(no_match_scores)])
print(f"\n{overlap} matches scored below average no-match score (concerning!)")
```

---

### Compare Multiple Thresholds

Test a range to find optimal:

```python
import json
from src.evaluation.evaluate_fast import evaluate_at_threshold

with open("data/eval/pair_scores.json") as f:
    scores_data = json.load(f)

scores_with_labels = [
    (item['job_id'], item['candidate_id'], 
     item['ground_truth_label'], item['semantic_score'])
    for item in scores_data
]

# Test range
thresholds = [0.55, 0.60, 0.65, 0.68, 0.70, 0.75, 0.80]

print("Threshold  Precision  Recall   F1-Score")
print("-" * 45)

for threshold in thresholds:
    metrics = evaluate_at_threshold(scores_with_labels, threshold)
    print(f"{threshold:.2f}       {metrics['precision']:.4f}    "
          f"{metrics['recall']:.4f}   {metrics['f1_score']:.4f}")
```

---

## 🚀 Next Steps & Iteration

### 1. Run Initial Evaluation

```bash
# Get baseline metrics
python src/evaluation/evaluate_fast.py
python src/evaluation/evaluate_sklearn.py
```

---

### 2. Note Optimal Threshold

From the results, identify:
- Best threshold by F1-score
- Precision/recall trade-offs
- ROC-AUC and Average Precision scores

Use this threshold in production:
```bash
python scripts/match_candidate.py 19 0.65  # Your optimal threshold
```

---

### 3. Analyze Errors

```bash
# Visualize and understand failures
python src/evaluation/visualize_results.py
```

Review:
- False positives (why scored high?)
- False negatives (why scored low?)
- Score distribution overlap

---

### 4. Iterate & Improve

**If precision is low:**
- ✅ Improve search text quality (more relevant fields)
- ✅ Filter out generic terms
- ✅ Increase threshold

**If recall is low:**
- ✅ Enhance embeddings (include more context)
- ✅ Review search text completeness
- ✅ Decrease threshold

**If both are low:**
- ✅ Check embedding generation
- ✅ Verify Qdrant collections
- ✅ Review ground truth labels
- ✅ Consider fine-tuning or different embedding model

---

### 5. Re-evaluate After Changes

Always re-run evaluation after improvements:

```bash
# After improving search text
python scripts/build_search_text.py
python scripts/embed_jobs.py
python scripts/embed_candidates.py

# Re-evaluate
python src/evaluation/evaluate_sklearn.py
```

---

## 📈 Performance Benchmarks

### Target Metrics (Production-Ready)

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| **F1-Score** | 0.60 | 0.70 | 0.80+ |
| **ROC-AUC** | 0.70 | 0.80 | 0.90+ |
| **Precision** | 0.60 | 0.70 | 0.85+ |
| **Recall** | 0.60 | 0.70 | 0.85+ |
| **Avg Precision** | 0.65 | 0.75 | 0.85+ |

---

## 💡 Pro Tips

### Tip 1: Balance Precision vs Recall

**High-stakes scenarios** (e.g., executive search):
- Prioritize **precision** → Use higher threshold (0.75+)
- Better to miss some than recommend bad matches

**High-volume scenarios** (e.g., entry-level positions):
- Prioritize **recall** → Use lower threshold (0.60-0.65)
- Cast wider net, let recruiters filter

---

### Tip 2: Use sklearn for Professional Reports

When presenting to stakeholders:
```bash
python src/evaluation/evaluate_sklearn.py > evaluation_report.txt
```

sklearn metrics are:
- ✅ Industry-standard
- ✅ Well-documented
- ✅ Easy to explain to non-technical audiences

---

### Tip 3: Monitor in Production

Track metrics over time:
```python
# Save timestamp with results
import json
from datetime import datetime

results = evaluate_system()
results["timestamp"] = datetime.now().isoformat()

with open(f"data/eval/results_{datetime.now():%Y%m%d}.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

### Tip 4: Compare Before/After

Keep evaluation results when testing improvements:
```bash
# Before changes
python src/evaluation/evaluate_sklearn.py
cp data/eval/sklearn_results.json data/eval/baseline.json

# Make improvements...

# After changes
python src/evaluation/evaluate_sklearn.py
# Compare baseline.json vs sklearn_results.json
```

---

## ❓ FAQ

<details>
<summary><b>Q: What's the difference between evaluate_fast.py and evaluate_sklearn.py?</b></summary>

**evaluate_fast.py:**
- Custom implementation
- Detailed threshold analysis
- Saves individual pair scores
- Good for deep-dive analysis

**evaluate_sklearn.py:**
- Uses scikit-learn (industry standard)
- Professional metrics (ROC-AUC, Classification Report)
- Concise output
- Better for reporting

**Recommendation:** Run both! They complement each other.
</details>

<details>
<summary><b>Q: Should I trust ROC-AUC or F1-Score more?</b></summary>

**Use F1-Score for:**
- Picking optimal threshold
- Day-to-day system evaluation
- Balancing precision/recall

**Use ROC-AUC for:**
- Overall system quality (threshold-independent)
- Comparing different models
- Understanding discriminative power

Both are important!
</details>

<details>
<summary><b>Q: My ROC-AUC is high but F1-Score is low. Why?</b></summary>

This happens when:
- Good separation between classes overall
- But wrong threshold chosen
- Or imbalanced dataset

**Solution:** Focus on finding optimal threshold using F1-score.
</details>

<details>
<summary><b>Q: How often should I re-evaluate?</b></summary>

**Re-evaluate when:**
- ✅ Changing embedding model
- ✅ Modifying search text generation
- ✅ Adding new features
- ✅ After significant data updates
- ✅ Monthly in production (monitoring)
</details>

---

## 🎓 Further Reading

- 📚 [Precision and Recall - Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall)
- 📊 [ROC Curve Analysis - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- 🎯 [F1 Score Interpretation](https://en.wikipedia.org/wiki/F-score)
- 📈 [Classification Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

<div align="center">

**Built with 💻 and 📊 for data-driven recruitment excellence**

*Measure. Analyze. Improve. Repeat.* 🔄

</div>
