# Job-Candidate Semantic Matching

A semantic matching system that uses vector embeddings to match job postings with candidates based on their skills, experience, and descriptions. Built with Google Gemini embeddings, Qdrant vector database, and MongoDB.

## Features

- **Semantic Search**: Match candidates to jobs using AI-powered embeddings
- **Vector Database**: Fast similarity search using Qdrant
- **Flexible Matching**: Search jobs for candidates or candidates for jobs
- **Score Thresholds**: Filter results by similarity scores (0.0 - 1.0)
- **MongoDB Integration**: Load and update job/candidate data from MongoDB

## Architecture

```
MongoDB (Data Storage)
    ↓
Search Text Builder (Combines relevant fields)
    ↓
Google Gemini API (text-embedding-004)
    ↓
Qdrant Vector Store (Cosine similarity search)
    ↓
Matching Results
```

## Project Structure

```
job-candidate-semantic-matching/
├── src/
│   ├── db/
│   │   ├── data_loader.py          # MongoDB data loading
│   │   └── __init__.py
│   ├── embedding/
│   │   ├── gemini.py               # Google Gemini embeddings
│   │   └── __init__.py
│   └── vector_store/
│       ├── qdrant.py               # Qdrant operations
│       └── __init__.py
├── scripts/
│   ├── build_search_text.py        # Build searchable text fields
│   ├── embed_jobs.py               # Embed jobs into Qdrant
│   ├── embed_candidates.py         # Embed candidates into Qdrant
│   ├── match_candidate.py          # Find jobs for a candidate
│   └── match_job.py                # Find candidates for a job
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

### 3. Match Jobs and Candidates

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

- `qdrant-client` - Vector database client
- `pymongo` - MongoDB driver
- `pandas` - Data manipulation
- `google-generativeai` - Google Gemini API
- `python-dotenv` - Environment variable management

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.