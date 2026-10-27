"""
Pytest configuration and shared test fixtures.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api import app
from src.analytics import RetailAnalyticsEngine
from src.ml_models import RetailMLManager


@pytest.fixture(scope="session")
def test_client():
    """Returns a TestClient instance for FastAPI integration tests."""
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def analytics_engine():
    """Returns a singleton RetailAnalyticsEngine instance."""
    return RetailAnalyticsEngine()


@pytest.fixture(scope="session")
def ml_manager():
    """Returns a singleton RetailMLManager instance with models loaded/trained."""
    manager = RetailMLManager()
    manager.load_or_train_models()
    return manager


@pytest.fixture
def sample_vip_payload():
    """Returns sample payload that maps to a VIP transaction."""
    return {
        "age": 24,
        "gender": "Male",
        "product_category": "Electronics",
        "quantity": 3,
        "price_per_unit": 500.0,
    }


@pytest.fixture
def sample_budget_payload():
    """Returns sample payload that maps to a standard budget transaction."""
    return {
        "age": 45,
        "gender": "Female",
        "product_category": "Clothing",
        "quantity": 1,
        "price_per_unit": 25.0,
    }
