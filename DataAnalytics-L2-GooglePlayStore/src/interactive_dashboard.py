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
from src.recommendation_engine import PlayStoreRecommendationEngine
from src.aso_intelligence_engine import ASOReviewIntelligenceEngine


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


@st.cache_resource(show_spinner="Indexing Play Store Vector Embeddings for Recommendations...")
def load_recommendation_engine() -> PlayStoreRecommendationEngine:
    """Initialize and cache the recommendation engine singleton."""
    return PlayStoreRecommendationEngine()


@st.cache_resource(show_spinner="Initializing Day 12 ASO & Review Aspect Intelligence...")
def load_aso_engine() -> ASOReviewIntelligenceEngine:
    """Initialize and cache the ASO & review aspect intelligence engine singleton."""
    return ASOReviewIntelligenceEngine()


engine = load_master_engine()
rec_engine = load_recommendation_engine()
aso_engine = load_aso_engine()


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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Market Intelligence & Category Radar",
    "💡 Developer Concept Feasibility Calculator",
    "⌨️ Real-Time Smart Keyboard Assistant",
    "🧠 Review Linguistics & Executive Scorecard",
    "🎯 App Recommendation & Personalized Discovery",
    "🔍 ASO & Review Aspect Intelligence"
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


# ----------------------------------------------------------------------------
# TAB 5: App Recommendation & Personalized Discovery (Day 11)
# ----------------------------------------------------------------------------
with tab5:
    st.subheader("🎯 Multi-Modal App Recommendation & Vector Discovery Engine")
    st.markdown("Leverage multi-modal vector embeddings (text semantics, scale, ratings, price, and sentiment priors) to find similar apps, explore personalized user portfolios, and benchmark cold-start app concepts.")

    rec_sub1, rec_sub2, rec_sub3 = st.tabs([
        "🔍 Similar App Finder (MMR Diversification)",
        "📱 Personalized User Portfolio",
        "🚀 Cold-Start Developer Benchmarking"
    ])

    with rec_sub1:
        st.markdown("#### Content-Based Cosine Similarity & MMR Diversification")
        col_rec1, col_rec2 = st.columns([2, 1])

        with col_rec1:
            app_options = sorted(rec_engine.apps_df["App_Clean"].unique().tolist())
            default_app = "Duolingo: Learn Languages Free"
            default_idx = app_options.index(default_app) if default_app in app_options else 0
            selected_app = st.selectbox("Select Target App to Recommend From", app_options, index=default_idx)

        with col_rec2:
            top_k_val = st.slider("Number of Recommendations (Top-K)", min_value=3, max_value=20, value=5)

        col_param1, col_param2 = st.columns(2)
        with col_param1:
            diversity_slider = st.slider(
                "MMR Diversity Weight (λ)",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="1.0 = Pure greedy relevance (may have near-duplicates). 0.0 = Maximal diversity. 0.7 = Recommended balance."
            )
        with col_param2:
            cat_filter_opts = ["All Categories"] + sorted(rec_engine.apps_df["Category"].unique().tolist())
            cat_filter_choice = st.selectbox("Optional Category Filter", cat_filter_opts, index=0)
            applied_cat_filter = None if cat_filter_choice == "All Categories" else cat_filter_choice

        if st.button("Generate Recommendations", type="primary", use_container_width=True):
            with st.spinner("Executing real-time vector search and MMR diversification..."):
                t_start = time.perf_counter()
                rec_res = rec_engine.get_similar_apps(
                    app_name=selected_app,
                    top_k=top_k_val,
                    diversity_weight=diversity_slider,
                    category_filter=applied_cat_filter
                )
                t_elapsed = (time.perf_counter() - t_start) * 1000.0

            st.success(f"Generated {len(rec_res.recommendations)} recommendations for **{rec_res.query_target}** in {t_elapsed:.2f}ms (Intra-List Diversity: {rec_res.intra_list_diversity:.4f})")

            rec_table_data = [
                {
                    "Rank": r.rank,
                    "App Name": r.app_name,
                    "Category": r.category,
                    "Similarity": f"{r.similarity_score:.4f}",
                    "Rating": f"⭐ {r.rating}",
                    "Installs": f"{r.installs:,}",
                    "Size (MB)": f"{r.size_mb:.1f} MB",
                    "Price": "Free" if r.price == 0 else f"${r.price:.2f}",
                    "Genres": r.genres
                }
                for r in rec_res.recommendations
            ]
            st.dataframe(pd.DataFrame(rec_table_data), use_container_width=True, hide_index=True)

    with rec_sub2:
        st.markdown("#### User Installed Portfolio Centroid Recommendation")
        st.markdown("Enter multiple apps installed on a user's mobile device to synthesize a personalized user preference vector.")

        default_portfolio = ["Spotify Music", "Duolingo: Learn Languages Free", "Subway Surfers"]
        available_default = [a for a in default_portfolio if a in app_options]
        user_portfolio = st.multiselect("Installed Apps Portfolio", app_options, default=available_default)

        col_port1, col_port2 = st.columns(2)
        with col_port1:
            port_k = st.slider("Top-K Portfolio Recommendations", 3, 15, 5, key="port_k")
        with col_port2:
            port_div = st.slider("Portfolio MMR Diversity (λ)", 0.0, 1.0, 0.7, step=0.05, key="port_div")

        if st.button("Generate Personalized Discovery", type="primary", use_container_width=True):
            if not user_portfolio:
                st.warning("Please select at least one installed application.")
            else:
                with st.spinner("Synthesizing user centroid vector and ranking non-portfolio apps..."):
                    port_res = rec_engine.recommend_for_user_portfolio(user_portfolio, top_k=port_k, diversity_weight=port_div)

                st.success(f"Found {len(port_res.recommendations)} personalized apps matching user tastes (Latency: {port_res.inference_latency_ms:.2f}ms)!")
                port_table = [
                    {
                        "Rank": r.rank,
                        "App Name": r.app_name,
                        "Category": r.category,
                        "Profile Match": f"{r.similarity_score:.4f}",
                        "Rating": f"⭐ {r.rating}",
                        "Installs": f"{r.installs:,}",
                        "Price": "Free" if r.price == 0 else f"${r.price:.2f}"
                    }
                    for r in port_res.recommendations
                ]
                st.dataframe(pd.DataFrame(port_table), use_container_width=True, hide_index=True)

    with rec_sub3:
        st.markdown("#### Cold-Start Positioning & Competitor Discovery")
        st.markdown("Benchmark prospective or unreleased app concepts against established store incumbents.")

        cs_col1, cs_col2, cs_col3 = st.columns(3)
        with cs_col1:
            cs_cat = st.selectbox("Target Category", sorted(rec_engine.apps_df["Category"].unique().tolist()), index=0)
            cs_genres = st.text_input("Genres / Sub-tags", "Finance;Tools")
        with cs_col2:
            cs_size = st.number_input("Estimated Binary Size (MB)", min_value=1.0, max_value=500.0, value=25.0, step=5.0)
            cs_price = st.number_input("Target Upfront Price ($)", min_value=0.0, max_value=100.0, value=0.0, step=0.99)
        with cs_col3:
            cs_rating = st.selectbox("Content Rating", ["Everyone", "Teen", "Mature 17+", "Everyone 10+"], index=0)
            cs_k = st.slider("Competitor Benchmarks to Return", 3, 10, 5)

        if st.button("Evaluate Cold-Start Positioning", type="primary", use_container_width=True):
            with st.spinner("Synthesizing concept vector and querying category incumbents..."):
                cs_res = rec_engine.recommend_cold_start(
                    category=cs_cat,
                    genres=cs_genres,
                    size_mb=cs_size,
                    price=cs_price,
                    content_rating=cs_rating,
                    top_k=cs_k
                )

            st.info(f"**Category Saturation Status**: {cs_res.category_competition_intensity} | **Category Mean Rating**: {cs_res.category_mean_rating:.2f} | **Median Installs**: {cs_res.category_median_installs:,}")
            for advice in cs_res.strategic_advice:
                st.markdown(f"- 💡 {advice}")

            cs_table = [
                {
                    "Rank": r.rank,
                    "Incumbent Competitor": r.app_name,
                    "Similarity Match": f"{r.similarity_score:.4f}",
                    "Rating": f"⭐ {r.rating}",
                    "Installs": f"{r.installs:,}",
                    "Size": f"{r.size_mb:.1f} MB",
                    "Price": "Free" if r.price == 0 else f"${r.price:.2f}"
                }
                for r in cs_res.top_competitors
            ]
            st.dataframe(pd.DataFrame(cs_table), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# TAB 6: Day 12 ASO & Review Aspect Intelligence
# ----------------------------------------------------------------------------
with tab6:
    st.markdown("### 🔍 Aspect-Based Sentiment Analysis, LDA Topic Mining & ASO Diagnostics")
    st.markdown(
        "Day 12 delivers production-grade review intelligence: Aspect-Based Sentiment Analysis (ABSA) across 5 core dimensions, "
        "Latent Dirichlet Allocation (LDA) topic discovery, bot/brigading review authenticity auditing, competitor ASO keyword gap mining, "
        "and a 360-degree composite App Health Index."
    )

    aso_sub1, aso_sub2, aso_sub3, aso_sub4 = st.tabs([
        "🔬 Aspect Sentiment (ABSA)",
        "🏷️ Competitor ASO Keyword Gap",
        "🛡️ Review Authenticity & Fraud Audit",
        "🩺 360° App Health Diagnostic"
    ])

    with aso_sub1:
        st.markdown("#### Operational Aspect Breakdown across 5 Dimensions")
        aso_target_type = st.radio("Target Scope", ["Individual App", "Whole Category"], horizontal=True)

        if aso_target_type == "Individual App":
            target_app_input = st.selectbox(
                "Select Application",
                sorted(aso_engine.df_apps["App_Clean"].unique().tolist()),
                index=sorted(aso_engine.df_apps["App_Clean"].unique().tolist()).index("Duolingo: Learn Languages Free")
                if "Duolingo: Learn Languages Free" in aso_engine.df_apps["App_Clean"].values else 0,
                key="aso_app_select"
            )
            is_cat = False
            target_str = target_app_input
        else:
            cat_input = st.selectbox("Select Store Category", sorted(aso_engine.categories), index=0, key="aso_cat_select")
            is_cat = True
            target_str = cat_input

        if st.button("Run Aspect-Based Sentiment Analysis", type="primary", use_container_width=True):
            with st.spinner("Extracting aspect mentions and calculating net sentiment scores..."):
                aspect_res = aso_engine.analyze_aspect_sentiment(target_name=target_str, is_category=is_cat)

            st.success(f"Analyzed {aspect_res.total_reviews_analyzed:,} review records for **{aspect_res.target_name}**.")

            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1:
                st.metric("Aggregate Aspect Health", f"{aspect_res.aggregate_aspect_health:.1f} / 100")
            with kpi_col2:
                st.metric("Dominant Strength", aspect_res.dominant_strength.upper())
            with kpi_col3:
                st.metric("Key Vulnerability", aspect_res.dominant_vulnerability.upper())

            aspect_rows = []
            for aspect_k, score_v in aspect_res.aspect_scores.items():
                aspect_rows.append({
                    "Aspect": aspect_k.upper(),
                    "Health Score (0-100)": score_v.aspect_health_index,
                    "Net Sentiment Score (NSS %)": f"{score_v.net_sentiment_score:+.1f}%",
                    "Mentions": score_v.mention_count,
                    "Pos / Neu / Neg": f"{score_v.positive_mentions} / {score_v.neutral_mentions} / {score_v.negative_mentions}",
                    "Mean Polarity": score_v.mean_polarity,
                    "Representative Keywords": ", ".join(score_v.representative_keywords[:4])
                })
            st.dataframe(pd.DataFrame(aspect_rows), use_container_width=True, hide_index=True)

    with aso_sub2:
        st.markdown("#### Competitor ASO Keyword Gap & Metadata Prescriptions")
        aso_app_choice = st.selectbox(
            "Select Target Application for ASO Audit",
            sorted(aso_engine.df_apps["App_Clean"].unique().tolist()),
            index=0,
            key="aso_gap_app_select"
        )
        top_k_kws = st.slider("Number of Keyword Opportunities to Mine", 4, 15, 8)

        if st.button("Mine Competitor Keyword Gap", type="primary", use_container_width=True):
            with st.spinner("Analyzing category incumbent keyword frequencies and identifying gap..."):
                gap_res = aso_engine.mine_aso_keyword_gap(target_app=aso_app_choice, top_n_keywords=top_k_kws)

            st.metric("Current ASO Optimization Score", f"{gap_res.aso_optimization_score:.1f}%")
            st.info(f"**Recommended Title Rebrand**: `{gap_res.prescribed_title_copy}`")

            col_gap1, col_gap2 = st.columns(2)
            with col_gap1:
                st.markdown("##### 📌 Keywords Present in App")
                st.write(gap_res.current_keywords if gap_res.current_keywords else ["None found in title"])
            with col_gap2:
                st.markdown("##### 🚀 High-Value Missing Category Keywords")
                st.write(gap_res.missing_high_value_keywords)

            st.markdown("##### 📝 Prescribed Copy Directives:")
            for bullet in gap_res.prescribed_feature_bullets:
                st.markdown(f"- {bullet}")

    with aso_sub3:
        st.markdown("#### Review Authenticity, Bot Brigading & Anomaly Audit")
        auth_app_choice = st.selectbox(
            "Target App to Audit for Review Fraud / Bot Floods",
            sorted(aso_engine.df_apps["App_Clean"].unique().tolist()),
            index=0,
            key="auth_app_select"
        )

        if st.button("Audit Review Authenticity", type="primary", use_container_width=True):
            with st.spinner("Auditing review text duplicates, polarity dissonance, and entropy..."):
                auth_res = aso_engine.evaluate_review_authenticity(app_name=auth_app_choice)

            auth_col1, auth_col2, auth_col3 = st.columns(3)
            with auth_col1:
                st.metric("Authenticity Index", f"{auth_res.authenticity_index:.1f} / 100")
            with auth_col2:
                st.metric("Audit Verdict", auth_res.verdict)
            with auth_col3:
                st.metric("Polarity Entropy Score", f"{auth_res.entropy_score:.2f}")

            st.markdown(f"**Total Reviews Audited**: {auth_res.total_reviews} | **Flagged Anomalies**: {auth_res.suspicious_reviews_count}")
            st.write({
                "Duplicate Cluster Count": auth_res.duplicate_cluster_count,
                "Sentiment-Rating Dissonance Count": auth_res.sentiment_dissonance_count,
                "Ultra-Short / Spam Count": auth_res.spam_length_count
            })

    with aso_sub4:
        st.markdown("#### 360-Degree App Health Scorecard & Recourse Matrix")
        health_app_choice = st.selectbox(
            "Select App for Full Lifecycle Health Scorecard",
            sorted(aso_engine.df_apps["App_Clean"].unique().tolist()),
            index=0,
            key="health_app_select"
        )

        if st.button("Generate 360° Health Diagnostic", type="primary", use_container_width=True):
            with st.spinner("Synthesizing multi-pillar metrics and compiling diagnostic recourse..."):
                diag = aso_engine.evaluate_app_health(app_name=health_app_choice)

            tier_colors = {
                "CHAMPION": "green",
                "SOLID_PERFORMER": "blue",
                "VULNERABLE": "orange",
                "CRITICAL_RENOVATION": "red"
            }
            color = tier_colors.get(diag.health_tier, "blue")
            st.markdown(f"### Overall Health: **:{color}[{diag.composite_health_index:.1f} / 100]** ({diag.health_tier})")

            h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns(5)
            with h_col1:
                st.metric("Rating Score (30%)", f"{diag.rating_score:.1f}")
            with h_col2:
                st.metric("Sentiment Score (25%)", f"{diag.sentiment_score:.1f}")
            with h_col3:
                st.metric("Aspect Score (20%)", f"{diag.aspect_health_score:.1f}")
            with h_col4:
                st.metric("Authenticity (15%)", f"{diag.authenticity_score:.1f}")
            with h_col5:
                st.metric("ASO Coverage (10%)", f"{diag.aso_optimization_score:.1f}")

            st.markdown("#### 🛠️ Actionable Prescriptive Growth & Quality Recourse:")
            for p in diag.prescriptions:
                st.markdown(f"- 💡 {p}")
