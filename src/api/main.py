"""FastAPI application for earnings call alignment analysis."""

import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import alignment, analysis, backtest, companies
from src.config import get_settings
from src.db.session import init_db

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Earnings Call Analyser API")
    init_db()
    logger.info("Database initialized")

    yield

    # Shutdown
    logger.info("Shutting down Earnings Call Analyser API")


app = FastAPI(
    title="Earnings Call Q&A Alignment Analyser",
    description="""
    Analyse management Q&A alignment in earnings calls using NLP.

    **Key Features:**
    - Extract Q&A pairs from earnings call transcripts
    - Compute alignment scores using FinBERT embeddings
    - Classify questions by category (margins, guidance, competition, etc.)
    - Backtest alignment signals against stock returns

    **API Endpoints:**
    - `/api/analyse` - Analyse a new earnings call transcript
    - `/api/companies/{ticker}/alignment` - Get historical alignment scores
    - `/api/compare` - Compare alignment across multiple companies
    - `/api/backtest` - Run backtest analysis

    Based on research: [Chiang et al. 2025](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608365/full)
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
app.include_router(companies.router, prefix="/api/companies", tags=["Companies"])
app.include_router(alignment.router, prefix="/api/alignment", tags=["Alignment"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtest"])


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "Earnings Call Q&A Alignment Analyser",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


def run() -> None:
    """Run the FastAPI server."""
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run()
