# Earnings Call Q&A Alignment Analyzer

## Project Overview

PyTorch-based NLP system measuring management Q&A alignment in earnings calls. Based on [Chiang et al. 2025](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608365/full) research.

**Key insight**: Low alignment (evasive answers) = negative signal for the stock.

## Tech Stack

- **ML**: PyTorch 2.x, HuggingFace Transformers (ProsusAI/FinBERT)
- **API**: FastAPI
- **Dashboard**: Streamlit + Plotly
- **Database**: PostgreSQL + SQLAlchemy
- **Deployment**: Docker, Kubernetes (Minikube)

## Key Commands

```bash
# Run API
uvicorn src.api.main:app --reload

# Run dashboard
streamlit run src/dashboard/app.py

# Run tests
pytest tests/ -v

# Docker Compose
docker-compose up --build
```

## Project Structure

- `src/data/` - Transcript providers, SEC EDGAR client, parser
  - `transcript_provider.py` - Unified facade with multi-source fallback and caching
  - `earningscall_client.py` - **Primary source** (free: AAPL/MSFT, paid: 5000+ companies)
  - `finnhub_client.py` - Requires paid subscription for transcripts
  - `alphavantage_client.py` - Secondary source (25 calls/day)
  - `fmp_client.py` - Legacy source (deprecated for new users Aug 2025)
- `src/nlp/` - FinBERT embeddings, question classifier
- `src/model/` - AlignmentClassifier, training loop
- `src/api/` - FastAPI routes
- `src/db/` - SQLAlchemy models
- `src/dashboard/` - Streamlit app (supports API fetch, paste, file upload)

## Environment Variables

**No API key needed for Apple/Microsoft transcripts** (EarningsCall free tier).

For 5,000+ companies, get an API key:
- `EARNINGSCALL_API_KEY` - Primary (https://earningscall.biz)
- `FINNHUB_API_KEY` - Transcripts require paid subscription
- `FMP_API_KEY` - Legacy (only works for pre-Aug 2025 accounts)

Configure priority with `TRANSCRIPT_PROVIDERS=earningscall,finnhub,fmp`

See `.env.example` for all options.

## Development Notes

- Use British English in documentation
- Target >70% accuracy on alignment classification
- Backtest should show p < 0.05 for statistical significance
