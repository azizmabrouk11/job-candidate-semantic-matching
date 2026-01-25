"""
Script to match candidates to job postings using vector similarity search.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_store import QdrantStore


def main(candidate_id=None):

    store = QdrantStore()
    result = store.search_jobs_for_candidate(candidate_id,score_threshold=0.7)
    print(f"Found {len(result)} matching jobs for candidate {candidate_id}")
    for job in result:
        print(f"- Job title: {job['title']}, Score: {job['score']:.4f}") 
    
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python match_candidate.py <candidate_id>")
        print("Example: python match_candidate.py 1")
        sys.exit(1)
    
    candidate_id = int(sys.argv[1])
    main(candidate_id)
