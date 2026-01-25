"""
Script to match jobs to candidates using vector similarity search.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_store import QdrantStore


def main(job_id=None,threshold=None):

    store = QdrantStore()
    result = store.search_candidates_for_job(job_id,score_threshold=threshold)
    print(f"Found {len(result)} matching candidates for job {job_id}")
    for candidate in result:
        print(f"- Candidate name: {candidate['name']}, Score: {candidate['score']:.4f}") 
    
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python match_job.py <job_id> [score_threshold]")
        print("Example: python match_job.py 1 0.7")
        sys.exit(1)
    
    job_id = int(sys.argv[1])
    score_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else None
    main(job_id, score_threshold)