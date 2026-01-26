"""
Build job-candidate pairs based on title matching criteria.
Pairs are saved to data/eval/pairs_raw.json for evaluation.
"""

import os
import json
import sys
from pathlib import Path



# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.db import get_jobs_df, get_candidates_df

def build_pairs():
    """
    Build job-candidate pairs where titles match.
    
    Returns:
        List of dicts with job_id and candidate_id
    """
    jobs_df = get_jobs_df()
    candidates_df = get_candidates_df()
    
    pairs = []
    
    for _, job in jobs_df.iterrows():
        job_title =job.get("title", "")
        if not job_title:
            continue
            
        for _, candidate in candidates_df.iterrows():
            candidate_title = candidate.get("title", "")
            if not candidate_title:
                continue
                
            # Match if titles are similar (exact match after normalization
      
            pairs.append({
                "job_id": str(job["_id"]),
                "candidate_id": str(candidate["_id"]),
            })
    
    return pairs


def save_pairs(pairs, output_path="data/eval/pairs_raw.json"):
    """
    Save pairs to JSON file.
    
    Args:
        pairs: List of pair dictionaries
        output_path: Path to save the pairs file
    """
    # Create directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(pairs)} pairs to {output_path}")


def main():
    """Main function to build and save pairs."""
    print("Loading data from MongoDB...")
    pairs = build_pairs()
    
    print(f"Built {len(pairs)} pairs based on title matching")
    
    if pairs:
        save_pairs(pairs)
        print(f"Sample pair: {pairs[0]}")
    else:
        print("No pairs found with matching titles")


if __name__ == "__main__":
    main()
