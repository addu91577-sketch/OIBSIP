"""Google Play Store Analysis & NLP Analytics Package.

Comprehensive analytical suite covering Play Store market intelligence (Task 4)
and natural language processing autocomplete & autocorrect systems (Task 5).
"""

from src.data_cleaning import (
    clean_playstore_apps,
    clean_user_reviews,
    parse_size_to_mb,
    generate_cleaning_audit,
    execute_pipeline
)

from src.category_ratings_analysis import (
    compute_category_distribution,
    compute_rating_statistics,
    compute_category_ratings,
    compute_size_installs_correlation,
    compute_stratified_correlations,
    plot_category_distribution,
    plot_rating_distributions,
    plot_size_vs_installs,
    plot_stratified_correlations,
    generate_category_ratings_reports,
    generate_all_day3_reports
)

from src.pricing_sentiment_analysis import (
    compute_pricing_distribution,
    compute_category_monetization,
    get_top_revenue_apps,
    compute_sentiment_scores,
    compute_sentiment_metrics,
    compute_category_sentiment_summary,
    compute_market_opportunity_matrix,
    generate_developer_recommendations,
    plot_pricing_distributions,
    plot_category_revenue,
    plot_sentiment_analysis,
    plot_market_opportunity_bubble,
    create_interactive_plotly_dashboard,
    generate_all_day4_reports
)

__all__ = [
    # Data Cleaning (Day 2)
    "clean_playstore_apps",
    "clean_user_reviews",
    "parse_size_to_mb",
    "generate_cleaning_audit",
    "execute_pipeline",
    # Category & Ratings EDA (Day 3)
    "compute_category_distribution",
    "compute_rating_statistics",
    "compute_category_ratings",
    "compute_size_installs_correlation",
    "compute_stratified_correlations",
    "plot_category_distribution",
    "plot_rating_distributions",
    "plot_size_vs_installs",
    "plot_stratified_correlations",
    "generate_category_ratings_reports",
    "generate_all_day3_reports",
    # Pricing, Sentiment & Dashboard (Day 4)
    "compute_pricing_distribution",
    "compute_category_monetization",
    "get_top_revenue_apps",
    "compute_sentiment_scores",
    "compute_sentiment_metrics",
    "compute_category_sentiment_summary",
    "compute_market_opportunity_matrix",
    "generate_developer_recommendations",
    "plot_pricing_distributions",
    "plot_category_revenue",
    "plot_sentiment_analysis",
    "plot_market_opportunity_bubble",
    "create_interactive_plotly_dashboard",
    "generate_all_day4_reports",
]
