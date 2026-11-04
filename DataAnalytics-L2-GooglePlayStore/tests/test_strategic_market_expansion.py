"""Unit and integration test suite for Day 15 Strategic Market Expansion Engine.

Validates:
1. Engine initialization, data ingestion, indexing, and category baselines.
2. Herfindahl-Hirschman Index (HHI) mathematical properties, normalization, and bounds.
3. Gini coefficient inequality calculation on equal and skewed distributions.
4. Category concentration ratios (CR4, CR8) and regulatory status assignment.
5. Category contestability index and barrier to entry classifications.
6. Comprehensive 33-category market concentration dataframe integrity.
7. App-level Competitive Moat Score (CMS), moat tiers, and strategic archetypes.
8. Synthetic / unlisted app concept moat scoring with custom parameters.
9. Qualified catalog-wide competitive moat scorecards.
10. Pairwise audience cannibalization risk, genre/token overlaps, and displacement probabilities.
11. Purchasing Power Parity (PPP) localized pricing across all 10 global economic zones.
12. Price elasticity revenue uplift and Global Expansion Readiness Index (GERI).
13. Multi-model hybrid monetization simulation (Pure Paid, Freemium, Ad-Supported, Subscription).
14. Algorithmic RICE feature prioritization and roadmap ranking.
15. CSV intelligence report export and file integrity.
16. Publication-grade Figure 30 and Figure 31 generation and non-empty file size.
17. FastAPI REST microservice endpoints for Day 15 strategic expansion.
"""

import math
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.strategic_market_expansion_engine import (
    PlayStoreStrategicExpansionEngine,
    MarketConcentrationResult,
    CompetitiveMoatResult,
    CannibalizationRiskResult,
    PPPRegionalPricingResult,
    GlobalExpansionPlanResult,
    HybridMonetizationResult,
    RICERoadmapItem,
    PPP_REGIONAL_FACTORS,
    STANDARD_RICE_INITIATIVES
)
from src.production_api import app

_BASE_DIR = Path(__file__).resolve().parent.parent
_REPORTS_DIR = _BASE_DIR / "reports"
_FIGURES_DIR = _REPORTS_DIR / "figures"


@pytest.fixture(scope="module")
def strat_engine():
    """Module-scoped fixture providing an initialized PlayStoreStrategicExpansionEngine."""
    return PlayStoreStrategicExpansionEngine()


@pytest.fixture(scope="module")
def api_client():
    """Module-scoped FastAPI TestClient fixture."""
    return TestClient(app)


# ============================================================================
# 1. Engine Initialization & Baselines
# ============================================================================

def test_engine_initialization(strat_engine):
    """Test that engine initializes properly with datasets, indices, and category benchmarks."""
    assert strat_engine._apps_df is not None
    assert len(strat_engine._apps_df) >= 9000
    assert len(strat_engine._app_name_index) >= 9000
    assert len(strat_engine._category_groups) == 33
    assert len(strat_engine._category_benchmarks) == 33
    assert "GAME" in strat_engine._category_benchmarks
    assert strat_engine._category_benchmarks["GAME"]["total_installs"] > 1000000


# ============================================================================
# 2. Herfindahl-Hirschman Index (HHI) & Gini Mathematical Rigor
# ============================================================================

def test_hhi_mathematical_bounds(strat_engine):
    """Test HHI satisfies 0 <= HHI <= 10,000 and HHI* is bounded between 0.0 and 1.0."""
    res = strat_engine.compute_category_concentration("GAME")
    assert isinstance(res, MarketConcentrationResult)
    assert 0.0 <= res.hhi <= 10000.0
    assert 0.0 <= res.hhi_normalized <= 1.0
    assert res.total_apps > 500
    assert res.total_installs > 10000000


def test_hhi_synthetic_monopoly_and_competition(strat_engine):
    """Test theoretical edge cases: single monopoly app vs evenly distributed apps."""
    # Synthetic Monopoly (1 app with 100% installs)
    monopoly_df = pd.DataFrame({
        "App": ["SoloApp"],
        "Category": ["MONOPOLY"],
        "Installs": [1000000],
        "Rating": [4.5],
        "Reviews": [50000],
        "Price": [0.0],
        "Size_MB": [25.0]
    })
    mono_engine = PlayStoreStrategicExpansionEngine(apps_df=monopoly_df, auto_load=False)
    mono_res = mono_engine.compute_category_concentration("MONOPOLY")
    assert math.isclose(mono_res.hhi, 10000.0, rel_tol=1e-3)
    assert math.isclose(mono_res.hhi_normalized, 1.0, rel_tol=1e-3)
    assert mono_res.concentration_status == "HIGHLY_CONCENTRATED_OLIGOPOLY"

    # Synthetic Perfect Competition (10 equal apps)
    comp_df = pd.DataFrame({
        "App": [f"App_{i}" for i in range(10)],
        "Category": ["COMPETITIVE"] * 10,
        "Installs": [100000] * 10,
        "Rating": [4.0] * 10,
        "Reviews": [1000] * 10,
        "Price": [0.0] * 10,
        "Size_MB": [15.0] * 10
    })
    comp_engine = PlayStoreStrategicExpansionEngine(apps_df=comp_df, auto_load=False)
    comp_res = comp_engine.compute_category_concentration("COMPETITIVE")
    # 10 apps * 10%^2 = 10 * 100 = 1000
    assert math.isclose(comp_res.hhi, 1000.0, rel_tol=1e-2)
    assert comp_res.concentration_status == "UNCONCENTRATED_COMPETITIVE"
    assert math.isclose(comp_res.hhi_normalized, 0.0, abs_tol=1e-3)


def test_gini_coefficient_properties(strat_engine):
    """Test Gini coefficient bounds and sensitivity to inequality."""
    # Equal distribution -> Gini ~ 0
    equal_df = pd.DataFrame({
        "App": [f"App_{i}" for i in range(20)],
        "Category": ["EQUAL_CAT"] * 20,
        "Installs": [10000] * 20,
        "Rating": [4.0] * 20,
        "Reviews": [100] * 20,
        "Price": [0.0] * 20,
        "Size_MB": [10.0] * 20
    })
    eq_engine = PlayStoreStrategicExpansionEngine(apps_df=equal_df, auto_load=False)
    eq_res = eq_engine.compute_category_concentration("EQUAL_CAT")
    assert math.isclose(eq_res.gini_installs, 0.0, abs_tol=0.01)

    # Real category has high inequality (power-law distribution)
    game_res = strat_engine.compute_category_concentration("GAME")
    assert 0.70 <= game_res.gini_installs <= 0.99


# ============================================================================
# 3. Category Concentration & Contestability
# ============================================================================

def test_category_concentration_known_categories(strat_engine):
    """Test concentration results across known Play Store sectors."""
    books_res = strat_engine.compute_category_concentration("BOOKS_AND_REFERENCE")
    assert books_res.hhi >= 1500.0  # High concentration due to dominant readers
    assert books_res.concentration_status in ["MODERATELY_CONCENTRATED", "HIGHLY_CONCENTRATED_OLIGOPOLY"]

    beauty_res = strat_engine.compute_category_concentration("BEAUTY")
    assert beauty_res.total_apps < 100
    assert len(beauty_res.top_apps) <= 5


def test_cr4_and_cr8_properties(strat_engine):
    """Verify that CR4 <= CR8 <= 100% and top apps are sorted descending."""
    res = strat_engine.compute_category_concentration("COMMUNICATION")
    assert 0.0 <= res.cr4_share <= res.cr8_share <= 100.0
    shares = [app["market_share_pct"] for app in res.top_apps]
    assert shares == sorted(shares, reverse=True)


def test_contestability_index_bounds(strat_engine):
    """Test category contestability index is bounded and maps to valid barrier levels."""
    for cat in ["GAME", "TOOLS", "EVENTS"]:
        res = strat_engine.compute_category_concentration(cat)
        assert 0.0 <= res.contestability_index <= 100.0
        assert res.barrier_to_entry in ["LOW", "MODERATE", "HIGH", "EXTREME"]


def test_all_category_concentrations_dataframe(strat_engine):
    """Test full dataframe computation across all 33 categories."""
    df = strat_engine.compute_all_category_concentrations()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 33
    expected_cols = [
        "Category", "Total_Apps", "Total_Installs", "HHI", "HHI_Normalized",
        "Concentration_Status", "Gini_Installs", "CR4_Share_Pct", "CR8_Share_Pct",
        "Contestability_Index", "Barrier_To_Entry"
    ]
    for col in expected_cols:
        assert col in df.columns
    assert df["HHI"].isna().sum() == 0


# ============================================================================
# 4. App Competitive Moat & Strategic Archetypes
# ============================================================================

def test_app_moat_scoring_known_apps(strat_engine):
    """Test competitive moat score for prominent flagship applications."""
    subway = strat_engine.compute_app_moat("Subway Surfers")
    assert isinstance(subway, CompetitiveMoatResult)
    assert subway.app_name == "Subway Surfers"
    assert subway.category == "GAME"
    assert subway.installs >= 1000000000
    assert subway.moat_score >= 80.0
    assert subway.moat_tier == "WIDE_MOAT"
    assert subway.archetype == "MARKET_LEADER"
    assert subway.relative_rating_advantage >= 0.0


def test_app_moat_custom_parameters(strat_engine):
    """Test moat scoring with custom synthetic app parameters."""
    custom = {
        "category": "PRODUCTIVITY",
        "installs": 500000,
        "rating": 4.6,
        "reviews": 20000,
        "price": 2.99,
        "size_mb": 18.0
    }
    res = strat_engine.compute_app_moat("SyntheticNotePad", custom_params=custom)
    assert res.app_name == "SyntheticNotePad"
    assert res.category == "PRODUCTIVITY"
    assert 0.0 <= res.moat_score <= 100.0
    assert res.archetype in [
        "MARKET_LEADER", "AGGRESSIVE_CHALLENGER", "NICHE_SPECIALIST",
        "VULNERABLE_INCUMBENT", "MARGINAL_FOLLOWER"
    ]


def test_app_moat_archetype_assignment(strat_engine):
    """Verify distinct archetype assignments under different market conditions."""
    # Leader
    custom_leader = {"category": "GAME", "installs": 1000000000, "rating": 4.5, "reviews": 10000000}
    res_leader = strat_engine.compute_app_moat("LeaderConcept", custom_params=custom_leader)
    assert res_leader.archetype == "MARKET_LEADER"

    # Vulnerable Incumbent (high installs, low rating)
    custom_vuln = {"category": "TOOLS", "installs": 500000000, "rating": 3.7, "reviews": 5000000}
    res_vuln = strat_engine.compute_app_moat("VulnConcept", custom_params=custom_vuln)
    assert res_vuln.archetype == "VULNERABLE_INCUMBENT"


def test_all_app_moats_dataframe(strat_engine):
    """Verify compute_all_app_moats returns populated dataframe."""
    df = strat_engine.compute_all_app_moats(min_installs=10000000)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 50
    assert "Moat_Score" in df.columns
    assert "Archetype" in df.columns


# ============================================================================
# 5. Cannibalization Risk & Displacement Modeling
# ============================================================================

def test_cannibalization_risk_identical_or_related(strat_engine):
    """Test cannibalization risk between two related endless runner games."""
    res = strat_engine.evaluate_cannibalization_risk("Subway Surfers", "Temple Run 2")
    assert isinstance(res, CannibalizationRiskResult)
    assert res.category_overlap is True
    assert res.cannibalization_score >= 50.0
    assert res.displacement_probability > 0.10
    assert res.risk_level in ["MODERATE_OVERLAP", "HIGH_CANNIBALIZATION", "CRITICAL_PORTFOLIO_CONFLICT"]


def test_cannibalization_risk_disparate_apps(strat_engine):
    """Test cannibalization risk between two completely unrelated apps."""
    res = strat_engine.evaluate_cannibalization_risk("Subway Surfers", "Dropbox")
    assert res.category_overlap is False
    assert res.cannibalization_score < 45.0
    assert res.risk_level in ["MINIMAL_RISK", "MODERATE_OVERLAP"]


def test_cannibalization_displacement_probability(strat_engine):
    """Verify displacement probability is strictly bounded in [0.0, 1.0]."""
    res = strat_engine.evaluate_cannibalization_risk("Pou", "Candy Crush Saga")
    assert 0.0 <= res.displacement_probability <= 1.0


# ============================================================================
# 6. Global PPP Regional Pricing Arbitrage
# ============================================================================

def test_ppp_pricing_all_10_regions(strat_engine):
    """Test that optimize_ppp_pricing covers all 10 economic regions."""
    plan = strat_engine.optimize_ppp_pricing(base_price=4.99, app_name="Subway Surfers")
    assert isinstance(plan, GlobalExpansionPlanResult)
    assert len(plan.regional_pricings) == 10
    region_ids = [r.region_id for r in plan.regional_pricings]
    for expected_reg in PPP_REGIONAL_FACTORS.keys():
        assert expected_reg in region_ids


def test_ppp_elasticity_revenue_uplift(strat_engine):
    """Verify that lowering price to local purchasing power creates positive revenue uplift."""
    plan = strat_engine.optimize_ppp_pricing(base_price=6.99)
    assert plan.total_potential_revenue_uplift_pct >= 0.0
    for r in plan.regional_pricings:
        if r.ppp_factor < 1.0:
            assert r.elasticity_uplift_multiplier >= 1.0
            assert r.localized_price <= r.base_usd_price


def test_global_expansion_readiness_index(strat_engine):
    """Verify GERI score bounds, tiers, and launch order."""
    plan = strat_engine.optimize_ppp_pricing(base_price=4.99, app_name="Subway Surfers")
    assert 0.0 <= plan.geri_score <= 100.0
    assert plan.geri_tier in ["GLOBAL_READY", "EXPANSION_VIABLE", "LOCALIZATION_REQUIRED", "DOMESTIC_ONLY"]
    assert len(plan.recommended_launch_order) >= 3


# ============================================================================
# 7. Hybrid Monetization Architecture Optimizer
# ============================================================================

def test_hybrid_monetization_utility_vs_gaming(strat_engine):
    """Verify that utility categories prioritize subscriptions and gaming prioritizes IAP/ads."""
    custom_util = {"category": "PRODUCTIVITY", "installs": 1000000, "rating": 4.5, "price": 0.0}
    res_util = strat_engine.recommend_hybrid_monetization(custom_app=custom_util)
    assert res_util.recommended_primary_model == "HYBRID_SUBSCRIPTION"
    assert res_util.hybrid_split.get("subscription", 0.0) >= 0.50

    custom_game = {"category": "GAME", "installs": 5000000, "rating": 4.4, "price": 0.0}
    res_game = strat_engine.recommend_hybrid_monetization(custom_app=custom_game)
    assert res_game.recommended_primary_model == "FREEMIUM_IAP"
    assert res_game.hybrid_split.get("iap", 0.0) >= 0.40


def test_hybrid_monetization_split_sums_to_one(strat_engine):
    """Verify hybrid channel revenue split percentages sum to 1.0."""
    res = strat_engine.recommend_hybrid_monetization(app_name="Subway Surfers")
    total_split = sum(res.hybrid_split.values())
    assert math.isclose(total_split, 1.0, abs_tol=0.01)
    assert res.estimated_annual_arpu > 0.0
    assert res.estimated_annual_revenue_per_100k > 0.0


# ============================================================================
# 8. Quantitative RICE Strategic Product Roadmap
# ============================================================================

def test_rice_roadmap_ranking_and_bounds(strat_engine):
    """Test RICE roadmap items are ranked descending with valid positive scores."""
    items = strat_engine.generate_rice_product_roadmap(app_name="Subway Surfers")
    assert len(items) == len(STANDARD_RICE_INITIATIVES)
    scores = [it.rice_score for it in items]
    assert scores == sorted(scores, reverse=True)
    for it in items:
        assert isinstance(it, RICERoadmapItem)
        assert it.rice_score > 0.0
        assert it.priority_tier in ["P0_CRITICAL", "P1_HIGH_IMPACT", "P2_MEDIUM_TERM", "P3_LONG_TERM"]


# ============================================================================
# 9. Reports & Publication Visualizations
# ============================================================================

def test_reports_generation_and_csv_integrity(strat_engine):
    """Verify that analytical CSV reports are generated and have valid schema."""
    reports = strat_engine.export_all_reports()
    assert "category_concentration_hhi" in reports
    assert "global_ppp_benchmarks" in reports
    assert "app_moat_scorecard" in reports
    assert "strategic_monetization_playbook" in reports

    for name, path in reports.items():
        assert path.exists()
        assert path.stat().st_size > 100
        df = pd.read_csv(path)
        assert not df.empty


def test_publication_figures_generation(strat_engine):
    """Verify Figure 30 and Figure 31 exist and have non-empty file size."""
    f30 = _FIGURES_DIR / "30_market_concentration_hhi_and_competitive_moats.png"
    f31 = _FIGURES_DIR / "31_global_ppp_pricing_and_monetization_frontier.png"

    if not f30.exists() or not f31.exists():
        f30, f31 = strat_engine.generate_publication_figures()

    assert f30.exists()
    assert f30.stat().st_size > 50000
    assert f31.exists()
    assert f31.stat().st_size > 50000


# ============================================================================
# 10. FastAPI REST Microservice Endpoints
# ============================================================================

def test_api_market_concentration_endpoint(api_client):
    """Test POST /api/v1/strategy/market-concentration."""
    payload = {"category": "GAME"}
    resp = api_client.post("/api/v1/strategy/market-concentration", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] == "GAME"
    assert "hhi" in data
    assert "gini_installs" in data
    assert "contestability_index" in data


def test_api_moat_scorecard_endpoint(api_client):
    """Test POST /api/v1/strategy/moat-scorecard."""
    payload = {
        "app_name": "Subway Surfers",
        "category": "GAME",
        "installs": 1000000000,
        "rating": 4.5,
        "reviews": 27000000
    }
    resp = api_client.post("/api/v1/strategy/moat-scorecard", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["app_name"] == "Subway Surfers"
    assert data["moat_score"] >= 80.0
    assert "archetype" in data
    assert "strategic_recommendation" in data


def test_api_cannibalization_risk_endpoint(api_client):
    """Test POST /api/v1/strategy/cannibalization-risk."""
    payload = {
        "app_a": "Subway Surfers",
        "app_b": "Temple Run 2",
        "category_a": "GAME",
        "category_b": "GAME"
    }
    resp = api_client.post("/api/v1/strategy/cannibalization-risk", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["app_a"] == "Subway Surfers"
    assert data["app_b"] == "Temple Run 2"
    assert 0.0 <= data["cannibalization_score"] <= 100.0
    assert "displacement_probability" in data


def test_api_ppp_pricing_endpoint(api_client):
    """Test POST /api/v1/strategy/ppp-pricing-optimizer."""
    payload = {
        "base_price": 4.99,
        "app_name": "Minecraft",
        "category": "FAMILY"
    }
    resp = api_client.post("/api/v1/strategy/ppp-pricing-optimizer", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["base_price"] == 4.99
    assert len(data["regional_pricings"]) == 10
    assert "total_potential_revenue_uplift_pct" in data


def test_api_monetization_endpoint(api_client):
    """Test POST /api/v1/strategy/monetization-recommendation."""
    payload = {
        "app_name": "Duolingo",
        "category": "EDUCATION",
        "installs": 100000000,
        "rating": 4.7
    }
    resp = api_client.post("/api/v1/strategy/monetization-recommendation", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "recommended_primary_model" in data
    assert "estimated_annual_arpu" in data
    assert "hybrid_split" in data


def test_api_category_benchmarks_endpoint(api_client):
    """Test GET /api/v1/strategy/category-concentration-benchmarks."""
    # All categories
    resp = api_client.get("/api/v1/strategy/category-concentration-benchmarks")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 33

    # Filtered category
    resp_filtered = api_client.get("/api/v1/strategy/category-concentration-benchmarks?category=GAME")
    assert resp_filtered.status_code == 200
    cat_data = resp_filtered.json()
    assert cat_data["Category"] == "GAME"
    assert "HHI" in cat_data
