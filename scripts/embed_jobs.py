"""
embed the search_text field of all jobs in the database and store the embeddings 
in Qdrant vector store.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db import get_jobs_df
from src.embedding import embed_text
from src.vector_store import QdrantStore


def main():
    print("Loading jobs from MongoDB...")
    jobs_df = get_jobs_df()
    print(f"Loaded {len(jobs_df)} jobs")
    
    print("\nGenerating embeddings and storing in Qdrant...")
    store = QdrantStore()
    store.upsert_jobs(jobs_df, embed_text)
    print("Done!")


if __name__ == "__main__":
    main()
