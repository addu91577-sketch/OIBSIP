"""
Unit tests for the Customer Cohort Retention Dynamics & Survival Analysis Engine.
"""

import pytest
from src.cohort_retention import CohortRetentionSurvivalEngine


@pytest.fixture(scope="module")
def cohort_engine():
    return CohortRetentionSurvivalEngine()


def test_customer_survival_data_preparation(cohort_engine):
    df = cohort_engine._customer_survival_df
    assert df is not None
    assert len(df) == 1000
    assert "cohort_month" in df.columns
    assert "survival_days" in df.columns
    assert "churn_event" in df.columns
    assert set(df["churn_event"].unique()).issubset({0, 1})


def test_cohort_retention_matrix(cohort_engine):
    res = cohort_engine.compute_cohort_retention_matrix()
    assert res["total_cohorts"] >= 12
    assert res["total_customers"] == 1000
    assert len(res["benchmark_average_retention"]) > 0

    matrix = res["cohort_matrix"]
    assert len(matrix) >= 12

    # Check conservation and initial period
    for row in matrix:
        assert row["cohort_size"] > 0
        assert row["periods"]["Month_0"]["retention_pct"] == 100.0
        assert row["periods"]["Month_0"]["active_customers"] == row["cohort_size"]


def test_kaplan_meier_overall_curve(cohort_engine):
    res = cohort_engine.compute_kaplan_meier_survival()
    assert res["analysis_horizon_days"] == 365
    overall = res["overall_survival"]

    assert overall["sample_size"] == 1000
    assert overall["churned_count"] + overall["censored_count"] == 1000
    assert overall["median_survival_days"] > 0

    timeline = overall["timeline"]
    assert len(timeline) > 0
    # First point should be day 0 with prob 1.0
    assert timeline[0]["day"] == 0
    assert timeline[0]["survival_probability"] == 1.0

    # Probabilities must be monotonic non-increasing
    for i in range(len(timeline) - 1):
        assert timeline[i]["survival_probability"] >= timeline[i + 1]["survival_probability"]
        assert 0.0 <= timeline[i]["ci_lower_95"] <= timeline[i]["survival_probability"]
        assert timeline[i]["survival_probability"] <= timeline[i]["ci_upper_95"] <= 1.0


def test_kaplan_meier_segmentation(cohort_engine):
    res = cohort_engine.compute_kaplan_meier_survival(segment_by="gender")
    assert "segmented_curves" in res
    assert "Female" in res["segmented_curves"]
    assert "Male" in res["segmented_curves"]

    for gender in ["Female", "Male"]:
        curve = res["segmented_curves"][gender]
        assert curve["sample_size"] > 0
        assert len(curve["timeline"]) > 0


def test_log_rank_test(cohort_engine):
    res = cohort_engine.compute_log_rank_test(segment_col="gender")
    assert "chi_square_statistic" in res
    assert res["chi_square_statistic"] >= 0.0
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["degrees_of_freedom"] == 1
    assert isinstance(res["statistically_significant"], bool)


def test_log_rank_test_invalid_segment(cohort_engine):
    with pytest.raises(ValueError):
        cohort_engine.compute_log_rank_test(segment_col="invalid_column")


def test_detect_churn_hazard_cliffs(cohort_engine):
    res = cohort_engine.detect_churn_hazard_cliffs(interval_days=15)
    assert res["interval_window_days"] == 15
    assert res["median_hazard_rate"] > 0
    assert "hazard_curve" in res
    assert len(res["hazard_curve"]) > 0

    for pt in res["hazard_curve"]:
        assert pt["at_risk_count"] >= 0
        assert pt["churned_in_interval"] >= 0
        assert pt["hazard_rate"] >= 0.0


def test_calculate_retention_campaign_roi(cohort_engine):
    res = cohort_engine.calculate_retention_campaign_roi(
        intervention_cost_per_user=15.0,
        expected_churn_reduction_pct=0.20,
    )
    assert res["target_population"] > 0
    assert res["total_campaign_cost"] > 0
    assert res["projected_saved_customers"] > 0
    assert res["gross_revenue_preserved"] > 0
    assert res["verdict"] in ["Value Accretive (Profitable)", "Dilutive (Revise Spend)"]


def test_calculate_retention_campaign_roi_invalid(cohort_engine):
    with pytest.raises(ValueError):
        cohort_engine.calculate_retention_campaign_roi(expected_churn_reduction_pct=0.95)
