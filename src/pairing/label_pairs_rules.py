"""
Label job-candidate pairs using rule-based scoring.
Reads pairs from data/eval/pairs_raw.json and produces data/eval/pairs_labeled_rules.json
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.db import get_jobs_df, get_candidates_df
from src.evaluation.rule_engine import label_pair


def label_pairs(pairs_path="data/eval/pairs_raw.json", output_path="data/eval/pairs_labeled_rules.json"):
    """
    Label pairs using rule-based scoring.
    
    Args:
        pairs_path: Path to raw pairs JSON file
        output_path: Path to save labeled pairs
    """
    # Load pairs
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    # Load job and candidate data
    print("Loading data from MongoDB...")
    jobs_df = get_jobs_df()
    candidates_df = get_candidates_df()
    
    # Create lookup dictionaries
    jobs_dict = {str(row["_id"]): row for _, row in jobs_df.iterrows()}
    candidates_dict = {str(row["_id"]): row for _, row in candidates_df.iterrows()}
    
    labeled_pairs = []
    
    print(f"Labeling {len(pairs)} pairs...")
    for i, pair in enumerate(pairs):
        job_id = pair["job_id"]
        candidate_id = pair["candidate_id"]
        
        # Get job and candidate data
        job = jobs_dict.get(job_id)
        candidate = candidates_dict.get(candidate_id)
        
       
        
        # Calculate rule scores
        label = label_pair(candidate, job)
        
       
        
        labeled_pairs.append({
            "job_id": job_id,
            "candidate_id": candidate_id,
            "label_matched": int(label),
        })
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(pairs)} pairs...")
    
    # Save labeled pairs
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(labeled_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(labeled_pairs)} labeled pairs to {output_path}")
    
    # Print statistics
    matched_count = sum(p["label_matched"]==1 for p in labeled_pairs)
    print(f"Total matched pairs: {matched_count} / {len(labeled_pairs)}")
    


def main():
    """Main function to label pairs."""
    label_pairs()


if __name__ == "__main__":
    main()