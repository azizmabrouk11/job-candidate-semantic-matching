"""
Build search_text field for jobs and candidates in MongoDB.
This field combines relevant information for semantic search.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db import update_jobs_search_text, update_candidates_search_text


def main():
    """Build search_text field for all jobs and candidates."""
    print("Building search_text fields...")
    print("-" * 50)
    
    jobs_updated = update_jobs_search_text()
    print(f"Updated search_text for {jobs_updated} jobs")
    
    candidates_updated = update_candidates_search_text()
    print(f"Updated search_text for {candidates_updated} candidates")
    
    print("-" * 50)
    print(f"Total: {jobs_updated} jobs and {candidates_updated} candidates updated")


if __name__ == "__main__":
    main()
