"""
Retail Sales Analytics & Machine Learning Intelligence Suite.
"""

from src.analytics import RetailAnalyticsEngine
from src.ml_models import RetailMLManager
from src.recommendations import BusinessRecommendationEngine
from src.mlops_drift import DriftDetector, calculate_psi, calculate_categorical_psi
from src.segmentation import CustomerSegmentationEngine
from src.simulation import FinancialWhatIfSimulator

__version__ = "1.1.0"

__all__ = [
    "RetailAnalyticsEngine",
    "RetailMLManager",
    "BusinessRecommendationEngine",
    "DriftDetector",
    "calculate_psi",
    "calculate_categorical_psi",
    "CustomerSegmentationEngine",
    "FinancialWhatIfSimulator",
]
