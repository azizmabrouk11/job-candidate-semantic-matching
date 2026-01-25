"""
Script to match candidates to job postings using vector similarity search.
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_store import QdrantStore


def main(candidate_id=None, score_threshold=None):

    store = QdrantStore()
    result = store.search_jobs_for_candidate(candidate_id,score_threshold=score_threshold)
    print(f"Found {len(result)} matching jobs for candidate {candidate_id}")
    for job in result:
        print(f"- Job title: {job['title']}, Score: {job['score']:.4f}") 
    
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python match_candidate.py <candidate_id> [score_threshold]")
        print("Example: python match_candidate.py 19 0.7")
        sys.exit(1)
    
    candidate_id = int(sys.argv[1])
    score_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else None
    main(candidate_id, score_threshold)