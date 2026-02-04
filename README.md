# Earnings Call Q&A Alignment Analyser

A PyTorch-based NLP system that measures how well management answers analyst questions during earnings calls. **Low alignment (evasive answers) = negative signal for the stock.**

Based on [Chiang et al. 2025](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608365/full) research analysing 192,000 earnings calls.

## Key Features

- **Q&A Extraction**: Parse earnings call transcripts to extract analyst questions and management answers
- **FinBERT Embeddings**: Domain-specific embeddings using ProsusAI/FinBERT
- **Alignment Classification**: PyTorch contrastive learning model to classify Q&A alignment
- **Question Categorisation**: Classify questions by topic (margins, guidance, competition, etc.)
- **Backtest Framework**: Test alignment signal against forward stock returns
- **Interactive Dashboard**: Streamlit visualisation with alignment timelines and sector comparisons

## Architecture

```
Transcript Ingestion → Q&A Parsing → FinBERT Encoding → Alignment Model → FastAPI → Streamlit Dashboard
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerised deployment)
- FMP API key (free tier: 250 requests/day)

### Local Development

1. Clone the repository:
```bash
cd earnings-call-analyser
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your FMP_API_KEY
```

5. Start PostgreSQL (using Docker):
```bash
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=earnings_analyser \
  postgres:15-alpine
```

6. Run the API:
```bash
uvicorn src.api.main:app --reload
```

7. Run the dashboard:
```bash
streamlit run src/dashboard/app.py
```

### Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Access:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - API Docs: http://localhost:8000/docs
```

### Kubernetes (Minikube)

```bash
# Start Minikube
minikube start

# Build images in Minikube's Docker
eval $(minikube docker-env)
docker build -f docker/Dockerfile.api -t earnings-analyser-api:latest .
docker build -f docker/Dockerfile.dashboard -t earnings-analyser-dashboard:latest .

# Apply manifests
kubectl apply -f k8s/

# Get dashboard URL
minikube service dashboard-service -n earnings-analyser --url
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyse` | POST | Analyse a new transcript |
| `/api/companies/{ticker}/alignment` | GET | Get alignment history |
| `/api/compare` | POST | Compare multiple companies |
| `/api/backtest/run` | POST | Run backtest analysis |
| `/api/alignment/rankings` | GET | Get companies ranked by alignment |
| `/api/alignment/sectors` | GET | Compare sectors |

See full API documentation at `/docs` when running the server.

## Project Structure

```
earnings-call-analyser/
├── src/
│   ├── data/           # FMP client, SEC client, transcript parser
│   ├── nlp/            # FinBERT embeddings, question classifier
│   ├── model/          # PyTorch AlignmentClassifier, training
│   ├── api/            # FastAPI routes
│   ├── db/             # SQLAlchemy models
│   └── dashboard/      # Streamlit pages
├── k8s/                # Kubernetes manifests
├── docker/             # Dockerfiles
├── tests/              # Test suite
├── docker-compose.yml
└── README.md
```

## Model Architecture

The alignment classifier uses contrastive learning:

```python
class AlignmentClassifier(nn.Module):
    """
    Projects question and answer embeddings to shared space,
    computes interaction features, and predicts alignment.
    """
    # Question/Answer projection heads
    # Interaction layer (concat + diff + product)
    # Classification head (3 classes)
    # Alignment score head (0-1 regression)
```

Training uses combined loss:
- Cross-entropy for classification
- MSE for alignment score regression
- Contrastive loss (SimCSE-inspired) for embedding space

## Data Sources

| Source | Data | Rate Limit |
|--------|------|------------|
| [Financial Modeling Prep](https://financialmodelingprep.com) | Earnings transcripts | 250 req/day (free) |
| [SEC EDGAR](https://www.sec.gov/edgar) | 8-K filings (backup) | Unlimited |
| [yfinance](https://pypi.org/project/yfinance/) | Stock prices | Reasonable |

## Configuration

Environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `FMP_API_KEY` | Financial Modeling Prep API key | Required |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://...` |
| `MODEL_DEVICE` | PyTorch device (`cpu`, `cuda`, `mps`) | `cpu` |
| `FINBERT_MODEL` | HuggingFace model name | `ProsusAI/finbert` |

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_alignment_classifier.py -v
```

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Model accuracy | >70% on alignment classification |
| Backtest signal | Statistically significant (p < 0.05) |
| Sharpe ratio | >0.8 for long/short strategy |
| Dashboard | Interactive, demo-ready |

## Technologies

- **ML**: PyTorch 2.x, HuggingFace Transformers
- **NLP**: ProsusAI/FinBERT, sentence-transformers
- **API**: FastAPI, Pydantic
- **Dashboard**: Streamlit, Plotly
- **Database**: PostgreSQL, SQLAlchemy
- **Deployment**: Docker, Kubernetes

## References

- [Chiang et al. 2025 - LLMs in Equity Markets](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608365/full)
- [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert)
- [SimCSE: Contrastive Learning of Sentence Embeddings](https://github.com/princeton-nlp/SimCSE)

## License

MIT
