"""
Retail Sales Analytics & Machine Learning Intelligence Suite.
"""

from src.analytics import RetailAnalyticsEngine
from src.ml_models import RetailMLManager
from src.recommendations import BusinessRecommendationEngine
from src.mlops_drift import DriftDetector, calculate_psi, calculate_categorical_psi
from src.segmentation import CustomerSegmentationEngine
from src.simulation import FinancialWhatIfSimulator
from src.elasticity_optimization import PriceElasticityOptimizer
from src.ab_experimentation import RetailABTestingEngine
from src.clv_churn import CustomerLifetimeValueEngine
from src.inventory_supply_chain import InventorySupplyChainEngine
from src.market_basket import MarketBasketAffinityEngine
from src.demand_forecasting import MultiHorizonDemandForecaster
from src.attribution_marketing import AttributionMarketingEngine
from src.gmroi_assortment import GMROIAssortmentEngine
from src.cohort_retention import CohortRetentionSurvivalEngine
from src.dynamic_pricing import DynamicPricingEngine
from src.next_best_action import NextBestActionEngine
from src.financial_risk import FinancialRiskStressEngine

__version__ = "1.5.0"

__all__ = [
    "RetailAnalyticsEngine",
    "RetailMLManager",
    "BusinessRecommendationEngine",
    "DriftDetector",
    "calculate_psi",
    "calculate_categorical_psi",
    "CustomerSegmentationEngine",
    "FinancialWhatIfSimulator",
    "PriceElasticityOptimizer",
    "RetailABTestingEngine",
    "CustomerLifetimeValueEngine",
    "InventorySupplyChainEngine",
    "MarketBasketAffinityEngine",
    "MultiHorizonDemandForecaster",
    "AttributionMarketingEngine",
    "GMROIAssortmentEngine",
    "CohortRetentionSurvivalEngine",
    "DynamicPricingEngine",
    "NextBestActionEngine",
    "FinancialRiskStressEngine",
]
