"""Google Play Store Analysis & NLP Analytics Package.

Comprehensive analytical suite covering Play Store market intelligence (Task 4)
and natural language processing autocomplete & autocorrect systems (Task 5).
"""

from src.data_cleaning import (
    clean_playstore_apps,
    clean_user_reviews,
    parse_size_to_mb,
    generate_cleaning_audit
)

__all__ = [
    "clean_playstore_apps",
    "clean_user_reviews",
    "parse_size_to_mb",
    "generate_cleaning_audit"
]
