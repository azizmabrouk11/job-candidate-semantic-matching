def label_pair(
    candidate,
    job,     
             
) -> bool :
    """
    Verifies that the candidate’s and job’s labels match using rules.
    Returns:
        - bool:   True if candidate's labels match job's labels
    """
    from src.rules import keyword_overlap
    from src.rules import experience 

    if not( experience(candidate, job, tolerance=1.0) and keyword_overlap(candidate, job, min_overlap_ratio=0.4, mode="jaccard") ):
        return False
    return True