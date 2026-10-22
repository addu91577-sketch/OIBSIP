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

__all__ = [
    "clean_playstore_apps",
    "clean_user_reviews",
    "parse_size_to_mb",
    "generate_cleaning_audit",
    "execute_pipeline",
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
    "generate_all_day3_reports"
]
