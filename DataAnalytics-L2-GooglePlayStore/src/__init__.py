"""Google Play Store Analysis & NLP Analytics Package.

Comprehensive analytical suite covering Play Store market intelligence (Task 4)
and natural language processing autocomplete & autocorrect systems (Task 5).
"""

from src.data_cleaning import clean_playstore_apps, clean_user_reviews
from src.category_ratings_analysis import (
    compute_category_distribution,
    compute_rating_statistics,
    compute_stratified_correlations
)
from src.pricing_sentiment_analysis import (
    compute_pricing_distribution,
    compute_category_monetization,
    compute_market_opportunity_matrix
)
from src.nlp_autocomplete_engine import (
    CorpusPreprocessor,
    NGramLanguageModel,
    AutocompleteEngine
)
from src.nlp_autocorrect_engine import (
    DistanceMetrics,
    NorvigCandidateGenerator,
    DamerauLevenshteinAutocorrectEngine,
    BayesianFrequencyAutocorrectEngine
)
from src.master_pipeline import GooglePlaystoreMasterEngine
from src.production_api import app, create_app
from src.delivery_showcase import ProjectDeliveryShowcase
from src.predictive_modeling import (
    PlayStorePredictiveEngine,
    calculate_psi,
    calculate_ks_statistic,
    calculate_wasserstein
)
from src.simulation_engine import (
    PlayStoreSimulationEngine,
    WhatIfScenarioResult,
    ABTestResult,
    PriceElasticityResult,
)
from src.recommendation_engine import (
    PlayStoreRecommendationEngine,
    AppRecommendationItem,
    RecommendationBatchResult,
    ColdStartRecommendationResult,
)
from src.aso_intelligence_engine import (
    ASOReviewIntelligenceEngine,
    AspectScore,
    AppAspectAnalysisResult,
    TopicCluster,
    ReviewAuthenticityReport,
    KeywordOpportunityItem,
    ASOGapAnalysisResult,
    AppHealthDiagnosticResult,
)

__all__ = [
    "clean_playstore_apps",
    "clean_user_reviews",
    "compute_category_distribution",
    "compute_rating_statistics",
    "compute_stratified_correlations",
    "compute_pricing_distribution",
    "compute_category_monetization",
    "compute_market_opportunity_matrix",
    "CorpusPreprocessor",
    "NGramLanguageModel",
    "AutocompleteEngine",
    "DistanceMetrics",
    "NorvigCandidateGenerator",
    "DamerauLevenshteinAutocorrectEngine",
    "BayesianFrequencyAutocorrectEngine",
    "GooglePlaystoreMasterEngine",
    "app",
    "create_app",
    "ProjectDeliveryShowcase",
    "PlayStorePredictiveEngine",
    "calculate_psi",
    "calculate_ks_statistic",
    "calculate_wasserstein",
    "PlayStoreSimulationEngine",
    "WhatIfScenarioResult",
    "ABTestResult",
    "PriceElasticityResult",
    "PlayStoreRecommendationEngine",
    "AppRecommendationItem",
    "RecommendationBatchResult",
    "ColdStartRecommendationResult",
    "ASOReviewIntelligenceEngine",
    "AspectScore",
    "AppAspectAnalysisResult",
    "TopicCluster",
    "ReviewAuthenticityReport",
    "KeywordOpportunityItem",
    "ASOGapAnalysisResult",
    "AppHealthDiagnosticResult"
]
