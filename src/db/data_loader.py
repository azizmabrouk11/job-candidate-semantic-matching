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


def build_job_search_text(job):
    """
    Build search_text field for a job by combining relevant fields.
    
    Args:
        job: Job document from MongoDB
        
    Returns:
        String containing combined searchable text
    """
    parts = []
    
    if job.get("title"):
        parts.append(job["title"])
    
    if job.get("location"):
        parts.append(job["location"])
    
    if job.get("required_skills"):
        skills = job["required_skills"]
        if isinstance(skills, list):
            parts.append(" ".join(skills))
        else:
            parts.append(str(skills))
    
    if job.get("description"):
        parts.append(job["description"])
    
    if job.get("experience_required"):
        parts.append(f"{job['experience_required']} years experience")
    
    return " ".join(parts)


def build_candidate_search_text(candidate):
    """
    Build search_text field for a candidate by combining relevant fields.
    
    Args:
        candidate: Candidate document from MongoDB
        
    Returns:
        String containing combined searchable text
    """
    parts = []
    
    if candidate.get("name"):
        parts.append(candidate["name"])
    
    if candidate.get("title"):
        parts.append(candidate["title"])
    
    if candidate.get("skills"):
        skills = candidate["skills"]
        if isinstance(skills, list):
            parts.append(" ".join(skills))
        else:
            parts.append(str(skills))
    
    if candidate.get("education"):
        parts.append(candidate["education"])
    
    if candidate.get("summary"):
        parts.append(candidate["summary"])
    
    if candidate.get("experience_years"):
        parts.append(f"{candidate['experience_years']} years experience")
    
    return " ".join(parts)


def update_jobs_search_text():
    """Update search_text field for all jobs in MongoDB."""
    jobs_collection = db[JOBS_COLLECTION]
    jobs = list(jobs_collection.find())
    
    updated_count = 0
    for job in jobs:
        search_text = build_job_search_text(job)
        jobs_collection.update_one(
            {"_id": job["_id"]},
            {"$set": {"search_text": search_text}}
        )
        updated_count += 1
    
    return updated_count


def update_candidates_search_text():
    """Update search_text field for all candidates in MongoDB."""
    candidates_collection = db[CANDIDATES_COLLECTION]
    candidates = list(candidates_collection.find())
    
    updated_count = 0
    for candidate in candidates:
        search_text = build_candidate_search_text(candidate)
        candidates_collection.update_one(
            {"_id": candidate["_id"]},
            {"$set": {"search_text": search_text}}
        )
        updated_count += 1
    
    return updated_count


if __name__ == "__main__":
    print("Jobs :", len(get_jobs_df()))
    print(get_jobs_df().head(2))
    print("\nCandidates :", len(get_candidates_df()))
    print(get_candidates_df().head(2))