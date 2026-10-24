"""Unit and integration tests for Day 6 Integrated Master Pipeline & Synthesis Module.
"""

from pathlib import Path
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.master_pipeline import GooglePlaystoreMasterEngine


@pytest.fixture(scope="module")
def master_engine():
    """Shared instance of GooglePlaystoreMasterEngine for test execution."""
    engine = GooglePlaystoreMasterEngine()
    return engine


def test_master_engine_initialization(master_engine):
    """Verify master engine properties, datasets and schema definitions."""
    assert master_engine.project_root.exists()
    assert master_engine.data_dir.exists()
    assert master_engine.reports_dir.exists()
    assert master_engine.figures_dir.exists()

    # Check apps dataframe
    apps = master_engine.apps_df
    assert isinstance(apps, pd.DataFrame)
    assert len(apps) == 9659
    assert "Category" in apps.columns
    assert "Rating" in apps.columns
    assert "Installs" in apps.columns

    # Check reviews dataframe
    revs = master_engine.reviews_df
    assert isinstance(revs, pd.DataFrame)
    assert len(revs) == 29692
    assert "Translated_Review" in revs.columns


def test_query_category_metrics_education(master_engine):
    """Verify category querying for a top opportunity sector (EDUCATION)."""
    metrics = master_engine.get_category_metrics("EDUCATION")

    assert metrics["Category"] == "EDUCATION"
    assert metrics["Total_Apps"] > 50
    assert metrics["Mean_Rating"] >= 4.0
    assert metrics["Opportunity_Score"] > 70.0
    assert "Delight" in metrics["Strategic_Quadrant"] or "Quadrant I" in metrics["Strategic_Quadrant"]
    assert isinstance(metrics["Strategic_Recommendation"], str)
    assert len(metrics["Strategic_Recommendation"]) > 20


def test_query_category_metrics_case_insensitivity(master_engine):
    """Verify case-insensitivity and formatting handling in category querying."""
    m_game = master_engine.get_category_metrics("game")
    assert m_game["Category"] == "GAME"
    assert m_game["Saturation_Status"] == "Hyper-Saturated (Red Ocean)"

    m_art = master_engine.get_category_metrics("art and design")
    assert m_art["Category"] == "ART_AND_DESIGN"


def test_query_category_invalid_raises_error(master_engine):
    """Verify querying an invalid category raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        master_engine.get_category_metrics("NON_EXISTENT_CATEGORY_XYZ")
    assert "not found" in str(excinfo.value)


def test_score_app_concept_evaluation(master_engine):
    """Verify developer app concept feasibility scoring logic across tiers."""
    # Concept 1: Free utility education app, 15MB
    concept_edu = master_engine.score_app_concept("EDUCATION", size_mb=15.0, price=0.0)
    assert 0.0 <= concept_edu["Feasibility_Score"] <= 100.0
    assert "Exceptional" in concept_edu["Feasibility_Tier"] or "Strong" in concept_edu["Feasibility_Tier"]

    # Concept 2: Expensive game ($19.99), 120MB in hyper-saturated category
    concept_game = master_engine.score_app_concept("GAME", size_mb=120.0, price=19.99)
    assert len(concept_game["Identified_Market_Risks"]) >= 1
    assert concept_game["Feasibility_Score"] < concept_edu["Feasibility_Score"]


def test_smart_keyboard_type_autocorrect_typo(master_engine):
    """Verify real-time autocorrect detection and correction of misspelled words."""
    result = master_engine.smart_keyboard_type("the new update is amzaing")

    assert result["is_typo"] is True
    assert result["active_word"] == "amzaing"
    assert result["autocorrected_word"] == "amazing"
    assert "amazing" in result["autocorrect_candidates"]
    assert len(result["top3_autocomplete_predictions"]) > 0
    assert result["inference_latency_ms"] < 250.0  # Real-time responsiveness constraint


def test_smart_keyboard_type_clean_word_retention(master_engine):
    """Verify correctly spelled words are preserved without false positive corruption."""
    result = master_engine.smart_keyboard_type("i love this game")

    assert result["is_typo"] is False
    assert result["active_word"] == "game"
    assert result["autocorrected_word"] == "game"


def test_compute_cross_task_synthesis(master_engine):
    """Verify cross-task synthesis merges catalog metadata and review linguistics."""
    cross_df, master_kpis = master_engine.compute_cross_task_synthesis()

    assert isinstance(cross_df, pd.DataFrame)
    assert len(cross_df) == 33  # All 33 Play Store categories represented
    assert "Lexical_Diversity_TTR" in cross_df.columns
    assert "Slang_Or_Typo_Pct" in cross_df.columns
    assert "Mean_VADER_Compound" in cross_df.columns

    # Verify master KPI table
    assert isinstance(master_kpis, pd.DataFrame)
    assert len(master_kpis) >= 15
    assert "Metric" in master_kpis.columns
    assert "Value" in master_kpis.columns


def test_all_day6_generated_artifacts_exist(master_engine):
    """Verify that all Day 6 reports and high-resolution figures exist on disk."""
    summary_path = master_engine.reports_dir / "master_synthesis_summary.csv"
    cross_path = master_engine.reports_dir / "cross_task_category_insights.csv"
    fig13_path = master_engine.figures_dir / "13_master_executive_scorecard.png"
    fig14_path = master_engine.figures_dir / "14_cross_task_sentiment_lexicon_synthesis.png"

    # If not yet generated, run pipeline once
    if not (summary_path.exists() and fig13_path.exists()):
        master_engine.run_master_pipeline()

    assert summary_path.exists(), f"Missing {summary_path}"
    assert cross_path.exists(), f"Missing {cross_path}"
    assert fig13_path.exists(), f"Missing {fig13_path}"
    assert fig14_path.exists(), f"Missing {fig14_path}"
    assert fig13_path.stat().st_size > 10000
    assert fig14_path.stat().st_size > 10000
