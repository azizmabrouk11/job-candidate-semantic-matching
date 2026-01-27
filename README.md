# Job-Candidate Semantic Matching

A hybrid matching system that combines semantic search with rule-based evaluation to match job postings with candidates. Built with Google Gemini embeddings, Qdrant vector database, MongoDB, and a custom rule engine for intelligent pair evaluation.

## Features

- **Semantic Search**: Match candidates to jobs using AI-powered embeddings
- **Rule-Based Evaluation**: Intelligent scoring using experience and keyword overlap rules
- **Pair Generation**: Automated job-candidate pair creation for evaluation
- **Vector Database**: Fast similarity search using Qdrant
- **Flexible Matching**: Search jobs for candidates or candidates for jobs
- **Score Thresholds**: Filter results by similarity scores (0.0 - 1.0)
- **MongoDB Integration**: Load and update job/candidate data from MongoDB
- **Evaluation Pipeline**: Generate and label pairs for testing matching quality

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer (MongoDB)                      │
│                                                              │
│  Jobs Collection              Candidates Collection         │
│  - title                      - name, title                 │
│  - required_skills            - skills                      │
│  - experience_required        - experience_years            │
│  - description                - education, summary          │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│              Text Processing & Embedding                     │
│                                                              │
│  Search Text Builder ──→ Google Gemini (text-embedding-004) │
│  (Combines relevant fields)     (768-dimensional vectors)   │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│           Vector Store (Qdrant) + Rule Engine               │
│                                                              │
│  ┌─────────────────┐        ┌──────────────────┐           │
│  │ Semantic Search │        │  Rule-Based      │           │
│  │ (Cosine)       │◄──────►│  Evaluation      │           │
│  │                 │        │  - Experience    │           │
│  │ Jobs Collection │        │  - Keyword Match │           │
│  │ Candidates Coll │        │  - Binary Labels │           │
│  └─────────────────┘        └──────────────────┘           │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│                   Matching Results                           │
│                                                              │
│  - Semantic similarity scores                               │
│  - Rule-based match labels (0/1)                            │
│  - Ranked candidate/job lists                               │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
job-candidate-semantic-matching/
├── src/
│   ├── db/
│   │   ├── data_loader.py          # MongoDB data loading & search text builders
│   │   └── __init__.py
│   ├── embedding/
│   │   ├── gemini.py               # Google Gemini embeddings API wrapper
│   │   └── __init__.py
│   ├── vector_store/
│   │   ├── qdrant.py               # Qdrant vector DB operations
│   │   └── __init__.py
│   ├── rules/
│   │   ├── experience.py           # Experience matching rule
│   │   ├── keyword_overlap.py      # Keyword/skill overlap rule
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── rule_engine.py          # Rule-based pair evaluation
│   ├── pairing/
│   │   ├── build_pairs.py          # Generate job-candidate pairs by title
│   │   └── label_pairs_rules.py    # Label pairs using rules
│   └── matching/                    # (Reserved for future matching logic)
├── scripts/
│   ├── build_search_text.py        # Build searchable text fields in MongoDB
│   ├── embed_jobs.py               # Embed jobs into Qdrant
│   ├── embed_candidates.py         # Embed candidates into Qdrant
│   ├── match_candidate.py          # Find jobs for a candidate
│   └── match_job.py                # Find candidates for a job
├── data/
│   └── eval/
│       ├── pairs_raw.json          # Raw job-candidate pairs
│       └── pairs_labeled_rules.json # Pairs with rule-based labels
├── notebooks/
│   ├── 01_embedding_pipeline.ipynb
│   └── 02_embedding_pipeline.ipynb
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- MongoDB instance with job and candidate data
- Google Gemini API key
- Qdrant vector database (local or cloud)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd job-candidate-semantic-matching
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create `.env` file**
   ```env
   # MongoDB Configuration
   MONGODB_URI=mongodb://localhost:27017/
   MONGO_DB_NAME=your_database_name
   MONGO_JOBS_COLLECTION=jobs
   MONGO_CANDIDATES_COLLECTION=candidates
   
   # Google Gemini API
   GEMINI_API_KEY=your_gemini_api_key
   
   # Qdrant Configuration (optional, defaults to localhost)
   QDRANT_URL=http://localhost:6333
   ```

4. **Start Qdrant** (if running locally with Docker)
   ```bash
   docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
   ```

## Usage

### Pipeline Overview

The system supports two main workflows:

**1. Semantic Matching Pipeline** (Vector Search)
- Build search text → Generate embeddings → Search by similarity

**2. Rule-Based Evaluation Pipeline**
- Generate pairs → Apply rules → Label matches

---

### 1. Build Search Text Fields

First, generate the `search_text` field for all jobs and candidates in MongoDB. This combines relevant fields (title, skills, description, experience, etc.) into a single optimized text field for embedding.

```bash
python scripts/build_search_text.py
```

**What it does:**
- Jobs: Combines `title`, `location`, `required_skills`, `description`, `experience_required`
- Candidates: Combines `name`, `title`, `skills`, `education`, `summary`, `experience_years`

### 2. Generate and Store Embeddings

#### Embed Jobs
```bash
python scripts/embed_jobs.py
```

This will:
- Load all jobs from MongoDB
- Generate embeddings using Google Gemini (text-embedding-004)
- Store vectors in Qdrant's `jobs` collection

#### Embed Candidates
```bash
python scripts/embed_candidates.py
```

This will:
- Load all candidates from MongoDB
- Generate embeddings using Google Gemini
- Store vectors in Qdrant's `candidates` collection

### 3. Semantic Matching (Vector Search)

#### Find Jobs for a Candidate
```bash
python scripts/match_candidate.py <candidate_id> [score_threshold]
```

**Examples:**
```bash
# Find top 5 matching jobs for candidate 19
python scripts/match_candidate.py 19

# Find jobs with similarity score >= 0.7
python scripts/match_candidate.py 19 0.7

# Find high-quality matches only
python scripts/match_candidate.py 19 0.85
```

#### Find Candidates for a Job
```bash
python scripts/match_job.py <job_id> [score_threshold]
```

**Examples:**
```bash
# Find top 5 matching candidates for job 15
python scripts/match_job.py 15

# Find candidates with similarity score >= 0.7
python scripts/match_job.py 15 0.7
```

### 4. Rule-Based Evaluation Pipeline

#### Generate Job-Candidate Pairs
```bash
python src/pairing/build_pairs.py
```

**What it does:**
- Loads jobs and candidates from MongoDB
- Creates pairs based on **title matching** (normalized comparison)
- Saves pairs to `data/eval/pairs_raw.json`

**Output format:**
```json
[
  {
    "job_id": "job_001",
    "candidate_id": "cand_042",
    "job_title": "Backend Developer",
    "candidate_title": "Backend Developer"
  }
]
```

#### Label Pairs with Rules
```bash
python src/pairing/label_pairs_rules.py
```

**What it does:**
- Reads pairs from `data/eval/pairs_raw.json`
- Applies rule-based evaluation:
  - **Experience Rule**: Candidate meets job's experience requirement (±1 year tolerance)
  - **Keyword Overlap Rule**: Jaccard similarity ≥ 0.4 between skills/titles
- Assigns binary labels (1 = match, 0 = no match)
- Saves to `data/eval/pairs_labeled_rules.json`

**Output format:**
```json
[
  {
    "job_id": "job_001",
    "candidate_id": "cand_042",
    "label_matched": 1
  }
]
```

**Rule Logic:**
- **Match (1)**: Both experience AND keyword overlap conditions are satisfied
- **No Match (0)**: At least one condition fails

---

### Understanding Scores

The system uses **cosine similarity** for matching:
- **1.0**: Perfect match (identical vectors)
- **0.8 - 1.0**: Excellent match
- **0.7 - 0.8**: Good match
- **0.6 - 0.7**: Moderate match
- **Below 0.6**: Weak match

**Recommended thresholds:**
- `0.7` - General matching (balanced results)
- `0.75` - Quality matching (fewer but better results)
- `0.8+` - High-precision matching (strict requirements)

## MongoDB Schema

### Jobs Collection
```json
{
  "_id": "job_001",
  "title": "Backend Developer",
  "required_skills": ["Node.js", "MongoDB", "REST"],
  "experience_required": 2,
  "location": "Remote",
  "description": "Backend developer needed...",
  "search_text": "Backend Developer\nRemote\nNode.js MongoDB REST\n..."
}
```

### Candidates Collection
```json
{
  "_id": "cand_001",
  "name": "John Doe",
  "title": "Backend Developer",
  "skills": ["Node.js", "MongoDB", "REST"],
  "experience_years": 3,
  "education": "Computer Science",
  "summary": "Backend developer with...",
  "search_text": "John Doe Backend Developer Node.js MongoDB..."
}
```

## API Components

### Database Module (`src.db`)
- `get_jobs_df()` - Load jobs as pandas DataFrame
- `get_candidates_df()` - Load candidates as pandas DataFrame
- `build_job_search_text(job)` - Build searchable text for a job
- `build_candidate_search_text(candidate)` - Build searchable text for a candidate
- `update_jobs_search_text()` - Update all jobs in MongoDB
- `update_candidates_search_text()` - Update all candidates in MongoDB

### Embedding Module (`src.embedding`)
- `GeminiEmbedder` - Gemini API wrapper class
- `get_embedder()` - Get singleton embedder instance
- `embed_text(text)` - Generate embedding for text (768 dimensions)

### Vector Store Module (`src.vector_store`)
- `QdrantStore` - Qdrant client wrapper
- `create_collection()` - Create Qdrant collection
- `upsert_jobs()` - Insert/update job embeddings
- `upsert_candidates()` - Insert/update candidate embeddings
- `search_jobs_for_candidate()` - Find matching jobs
- `search_candidates_for_job()` - Find matching candidates
- `get_candidate_vector()` - Retrieve candidate embedding
- `get_job_vector()` - Retrieve job embedding

### Rules Module (`src.rules`)

#### Experience Rule (`experience.py`)
```python
experience(candidate, job, tolerance=0.0) -> bool
```
Checks if candidate's years of experience meets or exceeds job's requirement (with optional tolerance).

**Parameters:**
- `candidate`: Candidate document with `experience_years`
- `job`: Job document with `experience_required`
- `tolerance`: Float (default 0.0), allows ±N years flexibility

**Returns:** `True` if candidate meets requirement

#### Keyword Overlap Rule (`keyword_overlap.py`)
```python
keyword_overlap(candidate, job, min_overlap_ratio=0.4, mode="overlap") -> bool
```
Computes keyword/skill similarity between candidate and job.

**Parameters:**
- `candidate`: Candidate document with `skills` and `title`
- `job`: Job document with `required_skills` and `title`
- `min_overlap_ratio`: Float (default 0.4), minimum similarity threshold
- `mode`: String - "overlap" (overlap coefficient), "jaccard", or "dice"

**Returns:** `True` if similarity meets threshold

**Similarity Modes:**
- **overlap**: `|intersection| / min(|A|, |B|)` - Best for asymmetric matching
- **jaccard**: `|intersection| / |union|` - Balanced similarity measure
- **dice**: `2 × |intersection| / (|A| + |B|)` - Harmonic mean similarity

### Evaluation Module (`src.evaluation`)

#### Rule Engine (`rule_engine.py`)
```python
label_pair(candidate, job) -> bool
```
Evaluates a job-candidate pair using combined rules.

**Logic:**
```
Match = experience(tolerance=1.0) AND keyword_overlap(min_ratio=0.4, mode="jaccard")
```

**Returns:** `True` (match) or `False` (no match)

### Pairing Module (`src.pairing`)

#### Pair Building (`build_pairs.py`)
- `build_pairs()` - Generate job-candidate pairs based on title matching
- `save_pairs()` - Save pairs to JSON file
- Creates pairs where normalized job titles match normalized candidate titles

#### Pair Labeling (`label_pairs_rules.py`)
- `label_pairs()` - Apply rule engine to all pairs
- Loads data from MongoDB, evaluates each pair, saves labels
- Outputs statistics on match rate

## Development

### Run Tests
```bash
python -m pytest tests/
```

### Explore Notebooks
```bash
jupyter notebook notebooks/
```

The notebooks contain exploratory analysis and pipeline demonstrations.

## Troubleshooting

### No Results with Low Threshold
If you get 0 results even with threshold 0:
- Check if embeddings exist: ensure you ran `embed_jobs.py` and `embed_candidates.py`
- Verify the ID exists in the collection
- Check Qdrant connection: visit `http://localhost:6333/dashboard`

### API Key Errors
- Ensure `GEMINI_API_KEY` is set in `.env`
- Verify API key is valid at [Google AI Studio](https://makersuite.google.com/app/apikey)

### MongoDB Connection Issues
- Check `MONGODB_URI` in `.env`
- Ensure MongoDB is running
- Verify database and collection names

## Dependencies

Core dependencies:
- `qdrant-client` - Vector database client for similarity search
- `pymongo` - MongoDB driver for data storage
- `pandas` - Data manipulation and analysis
- `google-generativeai` - Google Gemini API for embeddings
- `python-dotenv` - Environment variable management

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Evaluation & Testing

The project includes a comprehensive evaluation pipeline to assess matching quality:

### Evaluation Workflow

1. **Generate Pairs** - Create job-candidate pairs using title matching
2. **Apply Rules** - Label pairs using rule-based logic (creates ground truth)
3. **Run Evaluation** - Compare semantic matching against ground truth
4. **Analyze Results** - Review metrics and optimize thresholds

### Step-by-Step Evaluation

#### 1. Generate Ground Truth Labels (Already Done)
Your `data/eval/pairs_labeled_rules.json` contains 2502 labeled pairs - this is your ground truth.

#### 2. Run Evaluation
```bash
python src/evaluation/evaluate_fast.py
```

**What it does:**
- Loads ground truth labels from `pairs_labeled_rules.json`
- Fetches all embeddings from Qdrant
- Computes cosine similarity for each job-candidate pair
- Evaluates at multiple thresholds (0.5 - 0.9)
- Reports Precision, Recall, F1-Score, Accuracy
- Identifies optimal threshold

**Output:**
- `data/eval/evaluation_results.json` - Summary metrics
- `data/eval/pair_scores.json` - Individual pair scores

#### 3. Visualize Results
```bash
python src/evaluation/visualize_results.py
```

**What it shows:**
- Dataset statistics (positive/negative samples)
- Best performance metrics with confusion matrix
- Performance comparison across thresholds
- False positive/negative analysis
- Recommendations for threshold selection

### Understanding Evaluation Metrics

**Precision** = `TP / (TP + FP)`
- How many recommended matches are actually good?
- High precision = Few bad recommendations

**Recall** = `TP / (TP + FN)`
- How many good matches did we find?
- High recall = Don't miss good candidates

**F1-Score** = `2 × (Precision × Recall) / (Precision + Recall)`
- Balanced metric combining precision and recall
- Use this to find optimal threshold

**Accuracy** = `(TP + TN) / Total`
- Overall correctness

### Rule Configuration

Default rule thresholds (in `rule_engine.py`):
- Experience tolerance: `1.0 year`
- Keyword overlap ratio: `0.4` (40% Jaccard similarity)

Modify these in the rule functions for different matching strictness:

```python
# Stricter matching
label = label_pair(candidate, job)
# Uses: experience(tolerance=1.0) + keyword_overlap(min_ratio=0.4)

# More lenient
experience(candidate, job, tolerance=2.0)  # ±2 years
keyword_overlap(candidate, job, min_overlap_ratio=0.2)  # 20% overlap
```

### Interpreting Results

**Good System:**
- F1-Score > 0.75
- Precision & Recall both > 0.7
- Clear optimal threshold identified

**Needs Improvement:**
- F1-Score < 0.6
- Large gap between precision and recall
- Poor score distribution separation

**Action Items:**
- **Low Precision**: Increase threshold, improve embeddings
- **Low Recall**: Decrease threshold, enhance search text
- **Both Low**: Review embedding model or ground truth labels

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.