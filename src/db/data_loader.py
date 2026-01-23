# src/data_loader.py
"""
loading data from MongoDB and returning as pandas DataFrames.
"""

import os
import pandas as pd
from pymongo import MongoClient


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  



MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGO_DB_NAME")
JOBS_COLLECTION = os.getenv("MONGO_JOBS_COLLECTION")
CANDIDATES_COLLECTION = os.getenv("MONGO_CANDIDATES_COLLECTION")



client = MongoClient(MONGO_URI)
db = client[DB_NAME]


def get_jobs_df():
    """return all jobs as a DataFrame."""
    jobs = list(db[JOBS_COLLECTION].find())
    df = pd.DataFrame(jobs)
    if "_id" in df.columns:
        df["_id"] = df["_id"].astype(str)  # important pour Qdrant
    return df


def get_candidates_df():
    """return all candidates as a DataFrame."""
    candidates = list(db[CANDIDATES_COLLECTION].find())
    df = pd.DataFrame(candidates)
    if "_id" in df.columns:
        df["_id"] = df["_id"].astype(str)
    return df



if __name__ == "__main__":
    print("Jobs :", len(get_jobs_df()))
    print(get_jobs_df().head(2))
    print("\nCandidates :", len(get_candidates_df()))
    print(get_candidates_df().head(2))