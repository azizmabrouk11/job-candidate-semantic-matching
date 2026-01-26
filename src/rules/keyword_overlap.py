def keyword_overlap(
    candidate,
    job,     
    min_overlap_ratio: float = 0.4,
    mode: str = "overlap"         
) -> bool | float:
    """
    Compute keyword overlap between candidate and job.
    
    Returns:
        - bool:   True if overlap meets the threshold
        - float:  the actual similarity score (if you want the value)
    """
    # Build keyword lists
    cand_keywords = candidate.get("skills", [])
    if isinstance(cand_keywords, list):
        cand_keywords = cand_keywords.copy()  # Don't modify original
    else:
        cand_keywords = []
    
    if candidate.get("title"):
        cand_keywords.append(candidate.get("title"))
    
    job_keywords = job.get("required_skills", [])
    if isinstance(job_keywords, list):
        job_keywords = job_keywords.copy()
    else:
        job_keywords = []
        
    if job.get("title"):
        job_keywords.append(job.get("title"))

    # Convert to sets (removes duplicates & makes intersection fast)
    cand_set = set(k.lower().strip() for k in cand_keywords if k)
    job_set = set(k.lower().strip() for k in job_keywords if k)
    
    if not job_set:
        return False  # avoid division by zero
    
    intersection = cand_set & job_set
    overlap_size = len(intersection)
    
    if mode == "overlap":
        # Overlap coefficient = |intersection| / min(|A|, |B|)
        score = overlap_size / min(len(cand_set), len(job_set)) if min(len(cand_set), len(job_set)) > 0 else 0.0
    
    elif mode == "jaccard":
        # Jaccard similarity = |intersection| / |union|
        union = len(cand_set | job_set)
        score = overlap_size / union if union > 0 else 0.0
    
    elif mode == "dice":
        # Sørensen–Dice coefficient = 2 × |intersection| / (|A| + |B|)
        score = (2 * overlap_size) / (len(cand_set) + len(job_set)) if (len(cand_set) + len(job_set)) > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return score >= min_overlap_ratio 