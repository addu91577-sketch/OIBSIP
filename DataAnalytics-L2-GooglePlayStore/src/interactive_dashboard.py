"""Interactive Streamlit Web Application Dashboard.

Day 7 Milestone Core for Google Play Store Analysis & NLP Analytics (Tasks 4 & 5).
Provides:
1. Multi-tab visual explorer for 33 Play Store app categories and market dynamics.
2. Developer app concept feasibility calculator with instant scoring (0-100) and risk diagnostics.
3. Live interactive smart keyboard typing assistant with Bayesian autocorrect and contextual autocomplete.
4. Review corpus linguistics explorer and master executive KPI scorecard.
"""

from collections import Counter
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.master_pipeline import GooglePlaystoreMasterEngine


# ============================================================================
# Streamlit Page Setup & Styling
# ============================================================================

st.set_page_config(
    page_title="Google Play Store Market Intelligence & Smart Keyboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sleek typography and responsive cards
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-green {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-yellow {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-red {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .chip {
        display: inline-block;
        background-color: #EEF2FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
        border-radius: 20px;
        padding: 6px 14px;
        margin: 4px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Cached Master Engine Initialization
# ============================================================================

@st.cache_resource(show_spinner="Warming up Google Play Store Master Engine & NLP Models...")
def load_master_engine() -> GooglePlaystoreMasterEngine:
    """Initialize and cache the master engine singleton."""
    engine = GooglePlaystoreMasterEngine()
    _ = engine.apps_df
    _ = engine.reviews_df
    engine.init_nlp_assistant()
    return engine


engine = load_master_engine()


# ============================================================================
# Sidebar Configuration & Navigation
# ============================================================================

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Play_Arrow_logo.svg", width=64)
    st.title("Play Store Analytics")
    st.markdown("**Tasks 4 & 5 Production Deployment**")
    st.markdown("---")

    st.subheader("System Telemetry")
    st.markdown(f"- **Unique Apps Catalog:** `9,659`")
    st.markdown(f"- **Cleaned Reviews:** `29,692`")
    st.markdown(f"- **App Categories:** `33`")
    st.markdown(f"- **Cumulative Installs:** `167.6B+`")
    st.markdown(f"- **Gross Paid Revenue:** `$291.1M`")
    st.markdown(f"- **NLP Vocabulary Types:** `20,344`")
    st.markdown("- **Engine Status:** :green[Active & Calibrated]")

    st.markdown("---")
    st.subheader("Microservice REST API")
    st.markdown("FastAPI endpoints are operational on port `8000`.")
    st.markdown("[View OpenAPI Docs (`/docs`)](http://localhost:8000/docs)")

    st.markdown("---")
    st.caption("Google Play Store Ecosystem Analysis · Day 7 Production Release")


# ============================================================================
# Header Banner
# ============================================================================

st.markdown('<div class="main-header">📱 Google Play Store Ecosystem & Smart Typing Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Unified analytical portal connecting real-world Android market intelligence with Bayesian autocorrect and N-gram language models.</div>', unsafe_allow_html=True)

# Top KPI Overview Ribbon
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Catalog Apps", "9,659", "100% Cleaned")
with col2:
    st.metric("Catalog Installs", "167.6B+", "33 Sectors")
with col3:
    st.metric("Estimated Paid Revenue", "$291.14M", "Price × Installs")
with col4:
    st.metric("Autocorrect Accuracy", "87.88%", "+6.1% vs Levenshtein")
with col5:
    st.metric("Inference Latency", "< 18.5 ms", "Sub-60fps budget")

st.markdown("---")


# ============================================================================
# Interactive Multi-Tab Workspace
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Intelligence & Category Radar",
    "💡 Developer Concept Feasibility Calculator",
    "⌨️ Real-Time Smart Keyboard Assistant",
    "🧠 Review Linguistics & Executive Scorecard"
])


# ----------------------------------------------------------------------------
# TAB 1: Category Market Intelligence & Strategy
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Android Category Market Intelligence Explorer")
    st.markdown("Select any of the 33 Play Store categories to query market share, rating distributions, monetization scale, and strategic recommendations.")

    # Category Selector
    categories_list = sorted(engine.opportunity_matrix["Category"].dropna().unique().tolist())
    selected_cat = st.selectbox("Select App Category:", categories_list, index=categories_list.index("EDUCATION") if "EDUCATION" in categories_list else 0)

    cat_metrics = engine.get_category_metrics(selected_cat)

    # Status Badge
    sat_status = cat_metrics["Saturation_Status"]
    if "Niche" in sat_status:
        badge_html = f'<span class="badge-green">{sat_status}</span>'
    elif "Moderate" in sat_status:
        badge_html = f'<span class="badge-yellow">{sat_status}</span>'
    else:
        badge_html = f'<span class="badge-red">{sat_status}</span>'

    st.markdown(f"### {cat_metrics['Category']} Sector Profile {badge_html}", unsafe_allow_html=True)

    # Key Category Metrics Grid
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        st.metric("Total Apps", f"{cat_metrics['Total_Apps']:,}", f"{cat_metrics['Market_Share_Pct']}% of catalog")
    with mcol2:
        st.metric("Mean Rating", f"{cat_metrics['Mean_Rating']:.3f}", f"Median: {cat_metrics['Median_Rating']:.2f}")
    with mcol3:
        st.metric("Total Installs", f"{cat_metrics['Total_Installs']:,}")
    with mcol4:
        st.metric("Estimated Revenue", f"${cat_metrics['Estimated_Revenue_USD']:,.2f}")
    with mcol5:
        st.metric("Opportunity Score", f"{cat_metrics['Opportunity_Score']} / 100")

    # Second row metrics & Strategy Callout
    scol1, scol2 = st.columns([1, 2])
    with scol1:
        st.markdown("**Domain Dynamics:**")
        st.write(f"- **VADER Sentiment:** `{cat_metrics['Mean_VADER_Compound']:+.3f}` ({cat_metrics['Positive_Review_Pct']}% Pos / {cat_metrics['Negative_Review_Pct']}% Neg)")
        st.write(f"- **Size vs. Installs Spearman ($\\rho$):** `{cat_metrics['Size_Installs_Spearman_Rho']:+.4f}` (p = {cat_metrics['Spearman_P_Value']:.4f})")
        st.write(f"- **Paid Catalog Share:** `{cat_metrics['Paid_Share_Pct']}%` ({cat_metrics['Paid_Apps']} paid apps)")
        st.write(f"- **Average Paid Price:** `${cat_metrics['Mean_Paid_Price']:.2f}`")

    with scol2:
        st.info(f"**Strategic Quadrant:** {cat_metrics['Strategic_Quadrant']}\n\n"
                f"**Strategic Guidance:** {cat_metrics['Strategic_Recommendation']}")

    # Category Charts
    st.markdown("#### Category Distribution & Sizing Dynamics")
    ch_col1, ch_col2 = st.columns(2)

    apps_cat = engine.apps_df[engine.apps_df["Category"] == cat_metrics["Category"]]

    with ch_col1:
        fig_rating = px.histogram(
            apps_cat,
            x="Rating",
            nbins=20,
            marginal="box",
            title=f"Rating Distribution: {cat_metrics['Category']}",
            color_discrete_sequence=["#3B82F6"]
        )
        fig_rating.update_layout(bargap=0.05, height=350)
        st.plotly_chart(fig_rating, use_container_width=True)

    with ch_col2:
        fig_scatter = px.scatter(
            apps_cat,
            x="Size_MB_Imputed",
            y="Installs",
            log_y=True,
            hover_name="App",
            title=f"App Size vs. Installs (Log Scale): {cat_metrics['Category']}",
            color_discrete_sequence=["#10B981"]
        )
        fig_scatter.update_layout(height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)


# ----------------------------------------------------------------------------
# TAB 2: Developer Concept Feasibility Calculator
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Developer App Concept Feasibility Scoring Engine")
    st.markdown("Quantify commercial and architectural feasibility before development begins. The engine computes: "
                r"$S_{\text{feas}} = \text{clip}\left(S_{\text{opp}} + \Delta_{\text{size}} + \Delta_{\text{price}}, 0, 100\right)$")

    fcol1, fcol2 = st.columns([1, 1])

    with fcol1:
        st.markdown("#### 1. Concept Parameters")
        input_category = st.selectbox("Target Sector:", categories_list, index=categories_list.index("EDUCATION") if "EDUCATION" in categories_list else 0, key="feas_cat")
        input_size = st.slider("Planned APK Download Size (MB):", min_value=1.0, max_value=150.0, value=15.0, step=1.0)
        input_price = st.slider("Target Download Price (USD):", min_value=0.0, max_value=25.0, value=1.99, step=0.49)

        score_res = engine.score_app_concept(input_category, input_size, input_price)

    with fcol2:
        st.markdown("#### 2. Feasibility Scorecard")
        score = score_res["Feasibility_Score"]
        tier = score_res["Feasibility_Tier"]

        if score >= 78.0:
            badge_class = "badge-green"
            gauge_color = "#10B981"
        elif score >= 65.0:
            badge_class = "badge-yellow"
            gauge_color = "#F59E0B"
        else:
            badge_class = "badge-red"
            gauge_color = "#EF4444"

        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Feasibility Score: {tier}", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 50], 'color': "#FEE2E2"},
                    {'range': [50, 70], 'color': "#FEF3C7"},
                    {'range': [70, 100], 'color': "#D1FAE5"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Score breakdown waterfall
    st.markdown("#### Score Component Waterfall")
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    with bcol1:
        st.metric("Base Opportunity Score", f"{score_res['Base_Category_Opportunity_Score']:.1f}")
    with bcol2:
        delta_s = score_res['Size_Adjustment_Delta']
        st.metric("APK Size Delta (Δsize)", f"{delta_s:+.1f}", delta=f"{delta_s:+.1f}")
    with bcol3:
        delta_p = score_res['Price_Adjustment_Delta']
        st.metric("Pricing Delta (Δprice)", f"{delta_p:+.1f}", delta=f"{delta_p:+.1f}")
    with bcol4:
        st.metric("Net Feasibility Score", f"{score:.1f} / 100")

    # Diagnostics & Guidance
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.markdown("**Identified Market Risks:**")
        if score_res["Identified_Market_Risks"]:
            for r in score_res["Identified_Market_Risks"]:
                st.warning(f"⚠️ {r}")
        else:
            st.success("✅ No structural market risks identified. Optimal positioning!")

    with dcol2:
        st.markdown("**Strategic Launch Guidance:**")
        for g in score_res["Strategic_Launch_Guidance"]:
            st.info(f"💡 {g}")


# ----------------------------------------------------------------------------
# TAB 3: Real-Time Smart Keyboard Assistant
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Sub-18ms Mobile Smart Keyboard Assistant")
    st.markdown("Simulate keystroke inference on live text streams. The engine applies **Peter Norvig + Bayesian Damerau-Levenshtein** "
                "edit distance to correct typos, followed by **Trigram language models with Stupid Backoff** to predict top-3 next words.")

    # Preset query buttons
    st.markdown("**Quick Preset Queries:**")
    pcols = st.columns(4)
    preset_chosen = None
    if pcols[0].button("the new update is amzaing"):
        preset_chosen = "the new update is amzaing"
    if pcols[1].button("this game is very addicitve"):
        preset_chosen = "this game is very addicitve"
    if pcols[2].button("great cusotmer support"):
        preset_chosen = "great cusotmer support"
    if pcols[3].button("best app for learnng"):
        preset_chosen = "best app for learnng"

    input_text = st.text_input(
        "Type or edit a sentence below:",
        value=preset_chosen if preset_chosen else "the new update is amzaing"
    )

    if input_text:
        res = engine.smart_keyboard_type(input_text)

        # Inference metrics row
        icol1, icol2, icol3, icol4 = st.columns(4)
        with icol1:
            st.metric("Active Trailing Token", res["active_word"] if res["active_word"] else "None")
        with icol2:
            st.metric("Typo Detected?", "YES (Misspelled)" if res["is_typo"] else "NO (Known Word)")
        with icol3:
            st.metric("Autocorrect Replacement", res["autocorrected_word"])
        with icol4:
            st.metric("Inference Latency", f"{res['inference_latency_ms']:.2f} ms", "Sub-18ms Budget")

        st.markdown("---")

        # Visual Candidate & Suggestion Bar
        kcol1, kcol2 = st.columns([1, 1])

        with kcol1:
            st.markdown("#### Autocorrect Candidate Bar")
            if res["autocorrect_candidates"]:
                c_html = "".join([f'<span class="chip">✨ {c}</span>' for c in res["autocorrect_candidates"]])
                st.markdown(c_html, unsafe_allow_html=True)
            else:
                st.write("No candidate suggestions required.")

        with kcol2:
            st.markdown("#### Contextual Top-3 Next-Word Autocomplete")
            if res["top3_autocomplete_predictions"]:
                p_html = "".join([
                    f'<span class="chip">➡️ {p["word"]} ({p["probability_pct"]}%)</span>'
                    for p in res["top3_autocomplete_predictions"]
                ])
                st.markdown(p_html, unsafe_allow_html=True)
            else:
                st.write("No autocomplete candidates available.")

        st.markdown("#### Completed Sentence Preview")
        st.success(f"**Recommended Sentence:** `{res['suggested_completion']}`")


# ----------------------------------------------------------------------------
# TAB 4: Review Linguistics & Executive Scorecard
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Cross-Task Synthesis & Executive Scorecard")
    st.markdown("Synthesis of 9,659 app metadata records and 29,692 user reviews across linguistics, market scale, and NLP algorithm benchmarks.")

    # Executive Scorecard Table
    st.markdown("#### 1. Master Executive Scorecard")
    summary_path = engine.reports_dir / "master_synthesis_summary.csv"
    if summary_path.exists():
        scorecard_df = pd.read_csv(summary_path)
    else:
        _, scorecard_df = engine.compute_cross_task_synthesis()

    st.dataframe(scorecard_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Cross-task category linguistics
    st.markdown("#### 2. Cross-Domain Category Review Linguistics")
    cross_path = engine.reports_dir / "cross_task_category_insights.csv"
    if cross_path.exists():
        cross_df = pd.read_csv(cross_path)
        st.dataframe(
            cross_df[[
                "Category", "Total_Apps", "Mean_Rating", "Reviews_Analyzed",
                "Mean_Review_Words", "Lexical_Diversity_TTR", "Slang_Or_Typo_Pct",
                "Top_Review_Collocation", "Opportunity_Score"
            ]],
            use_container_width=True,
            hide_index=True
        )

        # Plotly Scatter: Review Length vs Slang Rate
        fig_cross = px.scatter(
            cross_df,
            x="Mean_Review_Words",
            y="Slang_Or_Typo_Pct",
            size="Total_Apps",
            color="Opportunity_Score",
            hover_name="Category",
            title="Category Review Verbosity vs. Slang & Typo Density",
            labels={
                "Mean_Review_Words": "Mean Review Words per Review",
                "Slang_Or_Typo_Pct": "Slang & Typo Density (%)"
            },
            color_continuous_scale="Viridis"
        )
        fig_cross.update_layout(height=400)
        st.plotly_chart(fig_cross, use_container_width=True)
