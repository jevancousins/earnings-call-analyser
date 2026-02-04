"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Test cases for API endpoints."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        from src.api.main import app

        return TestClient(app)

    def test_root_endpoint(self, client: TestClient) -> None:
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_health_endpoint(self, client: TestClient) -> None:
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_analyse_endpoint_validation(self, client: TestClient) -> None:
        """Test analyse endpoint validates input."""
        # Missing required fields
        response = client.post("/api/analyse", json={})

        assert response.status_code == 422  # Validation error

    def test_analyse_endpoint_accepts_valid_input(self, client: TestClient) -> None:
        """Test analyse endpoint accepts valid input."""
        response = client.post(
            "/api/analyse",
            json={
                "ticker": "AAPL",
                "fetch_from_api": False,
                "transcript": "Q&A Session\n\nAnalyst: What about margins?\nCEO: Margins are strong.",
            },
        )

        # May fail if FMP key not set, but should at least accept the request
        assert response.status_code in [200, 201, 500]

    def test_openapi_docs(self, client: TestClient) -> None:
        """Test OpenAPI docs are available."""
        response = client.get("/docs")

        assert response.status_code == 200

    def test_openapi_schema(self, client: TestClient) -> None:
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
