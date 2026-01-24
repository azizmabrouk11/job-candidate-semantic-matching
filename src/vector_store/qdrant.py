# src/vector_store/qdrant.py
"""
Qdrant vector database operations for job-candidate matching.
Handles storing and querying embeddings for jobs and candidates.
"""

import os
from typing import List, Dict, Any, Optional

import pandas as pd
from qdrant_client import QdrantClient, models


class QdrantStore:
    """
    Manages Qdrant vector database operations for job and candidate embeddings.
    
    Example:
        store = QdrantStore()
        store.upsert_jobs(jobs_df, embed_func)
        matches = store.search_jobs_for_candidate(candidate_id, limit=5)
    """
    
    JOBS_COLLECTION = "jobs"
    CANDIDATES_COLLECTION = "candidates"
    
    def __init__(
        self,
        url: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ):
        """
        Initialize Qdrant client connection.
        
        Args:
            url: Full URL to Qdrant (e.g., "http://localhost:6333")
            host: Qdrant host (alternative to url)
            port: Qdrant port (alternative to url)
        """
        if url:
            self.client = QdrantClient(url=url)
        elif host and port:
            self.client = QdrantClient(host=host, port=port)
        else:
            # Default to localhost
            self.client = QdrantClient(url="http://localhost:6333")
    
    def get_collections(self) -> List[str]:
        """Get list of all collections in Qdrant."""
        collections = self.client.get_collections()
        return [col.name for col in collections.collections]
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 768,
        distance: models.Distance = models.Distance.COSINE,
        recreate: bool = False,
    ) -> None:
        """
        Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of the vectors (768 for Gemini text-embedding-004)
            distance: Distance metric (COSINE, EUCLID, DOT)
            recreate: If True, delete existing collection first
        """
        if recreate:
            self.client.delete_collection(collection_name=collection_name)
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )
    
    def upsert_jobs(
        self,
        jobs_df: pd.DataFrame,
        embed_func,
        text_column: str = "search_text",
        batch_size: int = 100,
    ) -> int:
        """
        Generate embeddings for jobs and store them in Qdrant.
        
        Args:
            jobs_df: DataFrame containing job data
            embed_func: Function to generate embeddings (text -> List[float])
            text_column: Column containing text to embed
            batch_size: Number of records to upsert at once
            
        Returns:
            Number of jobs processed
        """
        job_records = []
        
        for index, job in jobs_df.iterrows():
            text_embedding = embed_func(job[text_column])
            
            job_records.append(models.PointStruct(
                id=index,
                payload={
                    "title": job.get("title", ""),
                    "mongodb_id": str(job["_id"]),
                },
                vector=text_embedding,
            ))
            
            # Batch upsert
            if len(job_records) >= batch_size:
                self.client.upsert(
                    collection_name=self.JOBS_COLLECTION,
                    points=job_records,
                )
                job_records = []
        
        # Upsert remaining records
        if job_records:
            self.client.upsert(
                collection_name=self.JOBS_COLLECTION,
                points=job_records,
            )
        
        return len(jobs_df)
    
    def upsert_candidates(
        self,
        candidates_df: pd.DataFrame,
        embed_func,
        text_column: str = "search_text",
        batch_size: int = 100,
    ) -> int:
        """
        Generate embeddings for candidates and store them in Qdrant.
        
        Args:
            candidates_df: DataFrame containing candidate data
            embed_func: Function to generate embeddings (text -> List[float])
            text_column: Column containing text to embed
            batch_size: Number of records to upsert at once
            
        Returns:
            Number of candidates processed
        """
        candidate_records = []
        
        for index, candidate in candidates_df.iterrows():
            text_embedding = embed_func(candidate[text_column])
            
            candidate_records.append(models.PointStruct(
                id=index,
                payload={
                    "name": candidate.get("name", ""),
                    "title": candidate.get("title", ""),
                    "mongodb_id": str(candidate["_id"]),
                },
                vector=text_embedding,
            ))
            
            # Batch upsert
            if len(candidate_records) >= batch_size:
                self.client.upsert(
                    collection_name=self.CANDIDATES_COLLECTION,
                    points=candidate_records,
                )
                candidate_records = []
        
        # Upsert remaining records
        if candidate_records:
            self.client.upsert(
                collection_name=self.CANDIDATES_COLLECTION,
                points=candidate_records,
            )
        
        return len(candidates_df)
    
    def get_candidate_vector(self, candidate_id: int) -> List[float]:
        """
        Retrieve a candidate's embedding vector from Qdrant.
        
        Args:
            candidate_id: ID of the candidate
            
        Returns:
            Embedding vector
        """
        candidate_point = self.client.retrieve(
            collection_name=self.CANDIDATES_COLLECTION,
            ids=[candidate_id],
            with_vectors=True,
            with_payload=False,
        )
        return candidate_point[0].vector
    
    def get_job_vector(self, job_id: int) -> List[float]:
        """
        Retrieve a job's embedding vector from Qdrant.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Embedding vector
        """
        job_point = self.client.retrieve(
            collection_name=self.JOBS_COLLECTION,
            ids=[job_id],
            with_vectors=True,
            with_payload=False,
        )
        return job_point[0].vector
    
    def search_jobs_for_candidate(
        self,
        candidate_id: int,
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar jobs for a given candidate.
        
        Args:
            candidate_id: ID of the candidate
            limit: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dicts with job info and similarity scores
        """
        candidate_vector = self.get_candidate_vector(candidate_id)
        
        search_result = self.client.query_points(
            collection_name=self.JOBS_COLLECTION,
            query=candidate_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        
        results = []
        for point in search_result.points:
            results.append({
                "job_id": point.id,
                "mongodb_id": point.payload["mongodb_id"],
                "title": point.payload["title"],
                "score": point.score,
            })
        
        return results
    
    def search_candidates_for_job(
        self,
        job_id: int,
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar candidates for a given job.
        
        Args:
            job_id: ID of the job
            limit: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dicts with candidate info and similarity scores
        """
        job_vector = self.get_job_vector(job_id)
        
        search_result = self.client.query_points(
            collection_name=self.CANDIDATES_COLLECTION,
            query=job_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        
        results = []
        for point in search_result.points:
            results.append({
                "candidate_id": point.id,
                "mongodb_id": point.payload["mongodb_id"],
                "name": point.payload.get("name", ""),
                "title": point.payload.get("title", ""),
                "score": point.score,
            })
        
        return results
    
    def search_by_text(
        self,
        text: str,
        embed_func,
        collection_name: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar items using a text query.
        
        Args:
            text: Query text
            embed_func: Function to generate embeddings
            collection_name: Collection to search in
            limit: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dicts with item info and similarity scores
        """
        query_vector = embed_func(text)
        
        search_result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        
        results = []
        for point in search_result.points:
            results.append({
                "id": point.id,
                "payload": point.payload,
                "score": point.score,
            })
        
        return results


def get_qdrant_store(url: str = "http://localhost:6333") -> QdrantStore:
    """
    Factory function to create a QdrantStore instance.
    
    Args:
        url: Qdrant URL (can be overridden by QDRANT_URL env var)
        
    Returns:
        QdrantStore instance
    """
    url = os.getenv("QDRANT_URL", url)
    return QdrantStore(url=url)
