"""
Executive Retail Sales Analytics & Machine Learning Intelligence Dashboard.

Built with Streamlit and Plotly to provide interactive business intelligence,
time-series decomposition, demographic segmentation, Pareto inequality analysis,
live real-time machine learning inference, continuous MLOps drift monitoring,
RFM customer behavioral micro-segmentation, and financial what-if simulation.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure src module is discoverable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analytics import RetailAnalyticsEngine
from src.ml_models import RetailMLManager
from src.recommendations import BusinessRecommendationEngine
from src.mlops_drift import DriftDetector
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

st.set_page_config(
    page_title="Retail Sales Analytics & Predictive Intelligence Suite",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_engines():
    analytics = RetailAnalyticsEngine()
    ml = RetailMLManager()
    ml.load_or_train_models()
    drift = DriftDetector(baseline_df=analytics.df)
    segmentation = CustomerSegmentationEngine()
    segmentation.fit_clusters()
    sim = FinancialWhatIfSimulator()
    elasticity = PriceElasticityOptimizer()
    ab_test = RetailABTestingEngine()
    clv = CustomerLifetimeValueEngine()
    inventory = InventorySupplyChainEngine()
    market_basket = MarketBasketAffinityEngine()
    forecasting = MultiHorizonDemandForecaster()
    attribution = AttributionMarketingEngine()
    gmroi = GMROIAssortmentEngine()
    cohort_retention = CohortRetentionSurvivalEngine()
    dynamic_pricing = DynamicPricingEngine()
    nba = NextBestActionEngine()
    financial_stress = FinancialRiskStressEngine()
    return (
        analytics, ml, drift, segmentation, sim, elasticity, ab_test, clv,
        inventory, market_basket, forecasting, attribution, gmroi, cohort_retention,
        dynamic_pricing, nba, financial_stress
    )


(
    analytics, ml, drift_engine, segmentation_engine, sim_engine, elasticity_engine,
    ab_engine, clv_engine, inventory_engine, market_basket_engine, forecasting_engine,
    attribution_engine, gmroi_engine, cohort_engine, dynamic_pricing_engine, nba_engine,
    financial_engine
) = load_engines()
df = analytics.df

# Custom CSS for dark modern executive theme
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🛍️ Retail Analytics & Machine Learning Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Production Decision Support System · Oasis Infobyte Data Analytics Internship · Level 1</div>',
    unsafe_allow_html=True,
)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("🔍 Cohort Slicing & Filters")

selected_categories = st.sidebar.multiselect(
    "Product Categories",
    options=sorted(df["Product Category"].unique()),
    default=sorted(df["Product Category"].unique()),
)

selected_genders = st.sidebar.multiselect(
    "Customer Gender",
    options=sorted(df["Gender"].unique()),
    default=sorted(df["Gender"].unique()),
)

age_range = st.sidebar.slider(
    "Customer Age Range",
    min_value=int(df["Age"].min()),
    max_value=int(df["Age"].max()),
    value=(int(df["Age"].min()), int(df["Age"].max())),
)

# Filter dataset
filtered_df = df[
    (df["Product Category"].isin(selected_categories))
    & (df["Gender"].isin(selected_genders))
    & (df["Age"].between(age_range[0], age_range[1]))
]

st.sidebar.markdown("---")
st.sidebar.info(
    f"📊 **Active Cohort Records**: {len(filtered_df):,} / {len(df):,} ({(len(filtered_df)/len(df))*100:.1f}%)"
)

# Tabs
(
    tab_overview,
    tab_timeseries,
    tab_demographics,
    tab_products,
    tab_pareto,
    tab_ml,
    tab_drift,
    tab_clusters,
    tab_whatif,
    tab_elasticity,
    tab_ab_test,
    tab_clv,
    tab_inventory,
    tab_market_basket,
    tab_forecasting,
    tab_attribution,
    tab_gmroi,
    tab_cohorts,
    tab_dynamic_pricing,
    tab_nba,
    tab_finance,
    tab_recs,
) = st.tabs(
    [
        "📊 Executive KPIs",
        "📈 Time-Series Trends",
        "👥 Customer Demographics",
        "🏷️ Merchandise & Pricing",
        "👑 Pareto Whale Economy",
        "🤖 AI / ML Simulator",
        "🛡️ MLOps Drift Guard",
        "🧬 RFM Micro-Segments",
        "🔮 What-If Simulator",
        "⚖️ Price Elasticity",
        "🧪 Bayesian A/B Testing",
        "💎 CLV & Churn Guard",
        "📦 Supply Chain & Inventory",
        "🛒 Market Basket Affinity",
        "⏱️ Multi-Horizon Forecasting",
        "📣 Marketing Attribution",
        "💎 GMROI & Assortment",
        "⏳ Cohort Retention & Survival",
        "🎯 Dynamic Bandit Pricing",
        "🎯 Causal Uplift & NBA",
        "🏦 Working Capital & Stress Test",
        "📋 Strategic Action Plan",
    ]
)

# ----------------- TAB 1: EXECUTIVE KPIS -----------------
with tab_overview:
    st.subheader("Storewide Commercial Performance Scorecard")

    rev = filtered_df["Total Amount"].sum()
    orders = len(filtered_df)
    units = filtered_df["Quantity"].sum()
    aov = rev / orders if orders else 0
    median_val = filtered_df["Total Amount"].median() if orders else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Revenue", f"${rev:,.2f}")
    with col2:
        st.metric("Total Orders", f"{orders:,}")
    with col3:
        st.metric("Units Sold", f"{units:,}")
    with col4:
        st.metric("Average Order Value (AOV)", f"${aov:,.2f}")
    with col5:
        st.metric("Median Spend", f"${median_val:,.2f}")

    st.markdown("---")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("##### 📅 Monthly Revenue Performance Trajectory")
        monthly_filt = (
            filtered_df.groupby("YearMonth")["Total Amount"].sum().reset_index()
        )
        fig_month = px.line(
            monthly_filt,
            x="YearMonth",
            y="Total Amount",
            markers=True,
            title="Monthly Revenue Trend ($)",
            labels={"YearMonth": "Month", "Total Amount": "Revenue ($)"},
            color_discrete_sequence=["#38bdf8"],
        )
        fig_month.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_month, use_container_width=True)

    with c2:
        st.markdown("##### 🛍️ Department Revenue Breakdown")
        cat_filt = (
            filtered_df.groupby("Product Category")["Total Amount"].sum().reset_index()
        )
        fig_cat = px.pie(
            cat_filt,
            values="Total Amount",
            names="Product Category",
            hole=0.45,
            color="Product Category",
            color_discrete_map={
                "Electronics": "#38bdf8",
                "Clothing": "#ec4899",
                "Beauty": "#a855f7",
            },
        )
        fig_cat.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_cat, use_container_width=True)

# ----------------- TAB 2: TIME-SERIES TRENDS -----------------
with tab_timeseries:
    st.subheader("Temporal Velocity & Shopping Cycles")

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("##### 🗓️ Quarterly Performance Comparison")
        q_summary = (
            filtered_df.groupby("Quarter")
            .agg(Revenue=("Total Amount", "sum"), Orders=("Transaction ID", "count"))
            .reset_index()
        )
        fig_q = px.bar(
            q_summary,
            x="Quarter",
            y="Revenue",
            text_auto="$,.0f",
            title="Quarterly Turnover ($)",
            color="Revenue",
            color_continuous_scale="Viridis",
        )
        fig_q.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig_q, use_container_width=True)

    with t_col2:
        st.markdown("##### ⚡ Day-of-Week Shopping Velocity")
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_df = (
            filtered_df.groupby("DayOfWeek")
            .agg(Revenue=("Total Amount", "sum"), Orders=("Transaction ID", "count"))
            .reindex(dow_order)
            .reset_index()
        )
        fig_dow = px.bar(
            dow_df,
            x="DayOfWeek",
            y="Revenue",
            title="Turnover by Day of Week ($)",
            color="DayOfWeek",
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig_dow.update_layout(template="plotly_dark", height=360, showlegend=False)
        st.plotly_chart(fig_dow, use_container_width=True)

# ----------------- TAB 3: CUSTOMER DEMOGRAPHICS -----------------
with tab_demographics:
    st.subheader("Demographic Profiling & Purchasing Behavior")

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown("##### 👥 Age Cohort Revenue Contribution")
        cohort_summary = (
            filtered_df.groupby("AgeCohort", observed=False)
            .agg(Revenue=("Total Amount", "sum"), AOV=("Total Amount", "mean"))
            .reset_index()
        )
        fig_cohort = px.bar(
            cohort_summary,
            x="AgeCohort",
            y="Revenue",
            text_auto="$,.0f",
            title="Turnover by Age Cohort ($)",
            color="AgeCohort",
            color_discrete_sequence=["#60a5fa", "#34d399", "#fbbf24", "#f87171"],
        )
        fig_cohort.update_layout(template="plotly_dark", height=360, showlegend=False)
        st.plotly_chart(fig_cohort, use_container_width=True)

    with d_col2:
        st.markdown("##### ⚖️ Gender Spending Parity")
        gender_summary = (
            filtered_df.groupby("Gender")
            .agg(Revenue=("Total Amount", "sum"), AOV=("Total Amount", "mean"))
            .reset_index()
        )
        fig_gender = px.bar(
            gender_summary,
            x="Gender",
            y="AOV",
            text_auto="$,.2f",
            title="Average Order Value by Gender ($)",
            color="Gender",
            color_discrete_map={"Female": "#ec4899", "Male": "#3b82f6"},
        )
        fig_gender.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig_gender, use_container_width=True)

# ----------------- TAB 4: MERCHANDISE & PRICING -----------------
with tab_products:
    st.subheader("Discrete Catalog Price Tiers & Category Architecture")

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("##### 🏷️ Revenue by Discrete Catalog Unit Price")
        price_tier_summary = (
            filtered_df.groupby("Price per Unit")
            .agg(Revenue=("Total Amount", "sum"), Volume=("Quantity", "sum"))
            .reset_index()
        )
        price_tier_summary["Price per Unit"] = price_tier_summary["Price per Unit"].astype(str) + " USD"
        fig_price = px.bar(
            price_tier_summary,
            x="Price per Unit",
            y="Revenue",
            text_auto="$,.0f",
            title="Turnover by Catalog Price Point ($)",
            color="Revenue",
            color_continuous_scale="Plasma",
        )
        fig_price.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig_price, use_container_width=True)

    with p_col2:
        st.markdown("##### 📦 Basket Unit Quantity Distribution")
        qty_summary = filtered_df["Quantity"].value_counts().sort_index().reset_index()
        qty_summary.columns = ["Quantity", "Order Count"]
        fig_qty = px.pie(
            qty_summary,
            values="Order Count",
            names="Quantity",
            title="Basket Unit Quantities (1 to 4 items)",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Agsunset,
        )
        fig_qty.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig_qty, use_container_width=True)

# ----------------- TAB 5: PARETO WHALE ECONOMY -----------------
with tab_pareto:
    st.subheader("The 30/80 Pareto Polarization & Lorenz Inequality Curve")

    pareto_data = analytics.get_pareto_concentration()
    gini_score = pareto_data["empirical_gini_index"]

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Empirical Gini Index (G)", f"{gini_score:.4f}", help="0 = Perfect Equality, 1 = Extreme Concentration")
    with kpi2:
        st.metric("Top 20% Orders Revenue Share", f"{pareto_data['top_20_percent_revenue_share_pct']}%")
    with kpi3:
        st.metric("Top 30% Orders Revenue Share", f"{pareto_data['top_30_percent_revenue_share_pct']}%")

    lorenz_pts = pd.DataFrame(pareto_data["lorenz_curve"])
    fig_lorenz = go.Figure()
    fig_lorenz.add_trace(
        go.Scatter(
            x=lorenz_pts["cumulative_orders_pct"],
            y=lorenz_pts["cumulative_revenue_pct"],
            mode="lines+markers",
            name="Empirical Lorenz Curve",
            line=dict(color="#38bdf8", width=3),
        )
    )
    fig_lorenz.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Line of Perfect Equality",
            line=dict(color="#94a3b8", dash="dash"),
        )
    )
    fig_lorenz.update_layout(
        title=f"Empirical Lorenz Curve of Spend Inequality (Gini = {gini_score:.4f})",
        xaxis_title="Cumulative Share of Orders (%)",
        yaxis_title="Cumulative Share of Revenue (%)",
        template="plotly_dark",
        height=420,
    )
    st.plotly_chart(fig_lorenz, use_container_width=True)

# ----------------- TAB 6: AI / ML SIMULATOR -----------------
with tab_ml:
    st.subheader("Interactive Predictive Machine Learning Simulator")
    st.caption("Test individual shopper profiles to generate real-time VIP Whale propensity scores and spend predictions.")

    c_in1, c_in2, c_in3, c_in4, c_in5 = st.columns(5)
    with c_in1:
        sim_age = st.number_input("Customer Age", min_value=18, max_value=85, value=28)
    with c_in2:
        sim_gender = st.selectbox("Gender", ["Female", "Male"])
    with c_in3:
        sim_cat = st.selectbox("Product Category", ["Electronics", "Clothing", "Beauty"])
    with c_in4:
        sim_qty = st.slider("Quantity (Units)", min_value=1, max_value=4, value=3)
    with c_in5:
        sim_price = st.selectbox("Price per Unit ($)", [25.0, 30.0, 50.0, 300.0, 500.0], index=3)

    if st.button("🚀 Run Real-Time AI Inference", use_container_width=True):
        vip_pred = ml.predict_vip(
            age=sim_age,
            gender=sim_gender,
            product_category=sim_cat,
            quantity=sim_qty,
            price_per_unit=sim_price,
        )
        spend_pred = ml.predict_spend(
            age=sim_age,
            gender=sim_gender,
            product_category=sim_cat,
            quantity=sim_qty,
            price_per_unit=sim_price,
        )
        bundle_rec = BusinessRecommendationEngine.generate_bundle(
            primary_category=sim_cat, target_units=sim_qty
        )

        st.markdown("---")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown("#### 👑 VIP Whale Propensity")
            st.metric("Probability", f"{vip_pred['propensity_score_pct']}%")
            st.info(f"**Classification Badge:** {vip_pred['tier_badge']}")

        with m_col2:
            st.markdown("#### 💵 Spend Regression")
            st.metric("Predicted Spend", f"${spend_pred['predicted_spend']:,.2f}")
            st.caption(
                f"95% Bounds: ${spend_pred['expected_range_min']:,.2f} – ${spend_pred['expected_range_max']:,.2f}"
            )

        with m_col3:
            st.markdown("#### 📦 Dynamic Bundle Lift")
            st.metric("Synergy Bundle Price", f"${bundle_rec['discounted_bundle_price']:,.2f}")
            st.success(f"Customer Saves: ${bundle_rec['customer_savings']:,.2f} ({bundle_rec['discount_percentage']}%)")

        with st.expander("🔍 View Algorithmic Bundle Details"):
            st.write(f"**Bundle Name:** {bundle_rec['bundle_name']}")
            st.write(f"**Upsell Impact:** {bundle_rec['business_upsell_value']}")
            st.table(pd.DataFrame(bundle_rec["bundle_items"]))

# ----------------- TAB 7: MLOPS DRIFT GUARD (DAY 9) -----------------
with tab_drift:
    st.subheader("Continuous MLOps Drift Monitoring & Data Integrity Guardrails")
    st.caption("Monitor Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) drift statistics against the training baseline.")

    col_drift_ctrl1, col_drift_ctrl2 = st.columns([2, 1])
    with col_drift_ctrl1:
        drift_scenario = st.selectbox(
            "Select Simulated Production Streaming Batch Scenario:",
            options=["stable", "tech_surge", "inflation", "youth_pivot"],
            format_func=lambda x: {
                "stable": "🟢 Stable Production Flow (Expected Baseline Behavior)",
                "tech_surge": "🔴 Tech & Luxury Surge (Heavy $500 Electronics Volume)",
                "inflation": "🟡 Price Inflation Shift (Across-the-board Price Escalation)",
                "youth_pivot": "🟡 Youth Influx (Young Adults 18-25 Purchasing Apparel)",
            }[x],
        )
    with col_drift_ctrl2:
        batch_size = st.slider("Live Batch Sample Size", min_value=50, max_value=500, value=250, step=50)

    # Generate batch and evaluate
    live_batch = drift_engine.generate_synthetic_batch(scenario=drift_scenario, n_samples=batch_size)
    drift_report = drift_engine.run_drift_audit(live_batch)

    d_card1, d_card2, d_card3, d_card4 = st.columns(4)
    with d_card1:
        st.metric("Global Drift Status", drift_report["global_status"])
    with d_card2:
        st.metric("Max Feature PSI", f"{drift_report['max_psi']:.4f}")
    with d_card3:
        st.metric("Critical Drift Features", f"{drift_report['critical_features_count']}")
    with d_card4:
        st.metric("Retraining Alert", "⚠️ REQUIRED" if drift_report["retrain_recommended"] else "✅ NOT NEEDED")

    if drift_report["retrain_recommended"]:
        st.error(f"🚨 **Automated MLOps Alert**: {drift_report['action_plan']}")
    else:
        st.success(f"🛡️ **System Healthy**: {drift_report['action_plan']}")

    st.markdown("---")
    st.markdown("##### 🔬 Feature-by-Feature Statistical Drift Metrics")
    feat_rows = []
    for feat, f_res in drift_report["feature_audits"].items():
        feat_rows.append({
            "Feature Name": feat,
            "Feature Type": f_res["type"],
            "PSI Score": f_res["psi"],
            "KS Statistic": f_res.get("ks_statistic", f_res.get("tvd", "-")),
            "P-Value": f_res.get("ks_p_value", "-"),
            "Alert Level": f_res["alert_level"],
            "Status": f_res["status"],
        })
    st.dataframe(pd.DataFrame(feat_rows), use_container_width=True)

    # Feature distribution comparison
    selected_drift_feat = st.selectbox(
        "Compare Feature Distributions (Baseline vs. Live Stream):",
        options=drift_engine.CONTINUOUS_FEATURES,
    )
    fig_drift_comp = go.Figure()
    fig_drift_comp.add_trace(
        go.Histogram(
            x=drift_engine.baseline_df[selected_drift_feat],
            name="Baseline Historical",
            opacity=0.6,
            marker_color="#38bdf8",
        )
    )
    fig_drift_comp.add_trace(
        go.Histogram(
            x=live_batch[selected_drift_feat],
            name="Live Production Batch",
            opacity=0.6,
            marker_color="#f43f5e",
        )
    )
    fig_drift_comp.update_layout(
        barmode="overlay",
        title=f"Distribution Comparison: {selected_drift_feat} (PSI: {drift_report['feature_audits'][selected_drift_feat]['psi']:.4f})",
        template="plotly_dark",
        height=380,
    )
    st.plotly_chart(fig_drift_comp, use_container_width=True)

# ----------------- TAB 8: RFM MICRO-SEGMENTS (DAY 9) -----------------
with tab_clusters:
    st.subheader("Customer Behavioral Micro-Segmentation & K-Means Clusters")
    st.caption("Unsupervised clustering on Recency, Monetary spend, Unit Price, Basket Quantity, and Customer Age.")

    seg_summary = segmentation_engine.get_summary()
    c_df = segmentation_engine.clustered_df

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Silhouette Score", f"{seg_summary['metrics']['silhouette_score']:.4f}", help="Higher is better (>0.35)")
    with sc2:
        st.metric("Davies-Bouldin Index", f"{seg_summary['metrics']['davies_bouldin_score']:.4f}", help="Lower is better (<1.0)")
    with sc3:
        st.metric("Optimal Clusters Identified", f"{seg_summary['metrics']['n_clusters']}")

    st.markdown("---")
    st.markdown("##### 💎 Customer Behavioral Cluster Profiles & Growth Playbook")

    for p in seg_summary["profiles"]:
        with st.container():
            st.markdown(f"### {p['archetype']} (Cluster {p['cluster_id']})")
            c_info1, c_info2, c_info3, c_info4 = st.columns(4)
            with c_info1:
                st.metric("Customer Base", f"{p['customer_count']:,} ({p['customer_share_pct']}%)")
            with c_info2:
                st.metric("Revenue Contribution", f"${p['total_revenue']:,.2f} ({p['revenue_share_pct']}%)")
            with c_info3:
                st.metric("Average Basket Spend", f"${p['mean_monetary']:,.2f}")
            with c_info4:
                st.metric("Avg Unit Price / Items", f"${p['mean_unit_price']:.0f} / {p['mean_quantity']:.1f} pcs")

            st.markdown(f"**Archetype Characterization:** {p['description']}")
            st.markdown(f"**🎯 Strategic Commercial Action:** `{p['recommended_strategy']}`")
            st.markdown("---")

    # 3D / 2D Scatter plot of clusters
    fig_scatter = px.scatter(
        c_df,
        x="Recency",
        y="Monetary",
        color="Cluster",
        size="Quantity",
        hover_data=["Customer ID", "Product Category", "Price per Unit", "Age"],
        title="Customer RFM Behavioral Clusters (Recency vs. Monetary Spend)",
        labels={"Recency": "Recency (Days since last order)", "Monetary": "Monetary Basket Spend ($)"},
        color_continuous_scale="Turbo",
    )
    fig_scatter.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------- TAB 9: WHAT-IF SIMULATOR (DAY 9) -----------------
with tab_whatif:
    st.subheader("Strategic What-If Scenario & Financial Uplift Simulator")
    st.caption("Simulate real-world interventions to break the 2.5-unit inelasticity cap and calculate projected ROI.")

    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    with w_col1:
        sim_bundle_adoption = st.slider("Bundle Adoption Rate (%)", min_value=5, max_value=60, value=25, step=5) / 100
    with w_col2:
        sim_target_units = st.slider("Target Basket Units", min_value=2.8, max_value=4.0, value=3.2, step=0.1)
    with w_col3:
        sim_bundle_discount = st.slider("Bundle Discount (%)", min_value=5, max_value=25, value=10, step=1) / 100
    with w_col4:
        sim_vip_lift = st.slider("VIP Whale Repeat Boost (%)", min_value=5, max_value=40, value=15, step=5) / 100

    sim_result = sim_engine.simulate_combined_strategy(
        bundle_adoption=sim_bundle_adoption,
        target_units=sim_target_units,
        bundle_discount=sim_bundle_discount,
        vip_lift=sim_vip_lift,
    )

    st.markdown("---")
    st.markdown("##### 📈 Projected Financial Return & Margin Analysis")

    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    with r_col1:
        st.metric(
            "Projected Total Turnover",
            f"${sim_result['projected_total_revenue']:,.2f}",
            delta=f"+${sim_result['total_incremental_revenue']:,.2f} (+{sim_result['overall_pct_revenue_lift']}%)",
        )
    with r_col2:
        st.metric("Net Profit Gain", f"${sim_result['total_net_profit_gain']:,.2f}")
    with r_col3:
        st.metric("Total Program Investment", f"${sim_result['total_program_cost']:,.2f}")
    with r_col4:
        st.metric("Program ROI", f"{sim_result['combined_roi_pct']:.1f}%")

    # Before vs. After Waterfall / Comparison
    waterfall_df = pd.DataFrame([
        {"Metric": "Baseline Revenue", "Amount": sim_result["baseline_revenue"]},
        {"Metric": "Bundle Expansion Lift", "Amount": sim_result["bundle_component"]["financial_impact"]["net_revenue_uplift"]},
        {"Metric": "VIP Retention Lift", "Amount": sim_result["vip_component"]["financial_impact"]["incremental_revenue"]},
        {"Metric": "Projected Total Revenue", "Amount": sim_result["projected_total_revenue"]},
    ])
    fig_fin = px.bar(
        waterfall_df,
        x="Metric",
        y="Amount",
        text_auto="$,.0f",
        title="Commercial Revenue Uplift Architecture ($)",
        color="Metric",
        color_discrete_sequence=["#64748b", "#38bdf8", "#a855f7", "#22c55e"],
    )
    fig_fin.update_layout(template="plotly_dark", height=380, showlegend=False)
    st.plotly_chart(fig_fin, use_container_width=True)

# ----------------- TAB 10: PRICE ELASTICITY & REVENUE OPTIMIZATION -----------------
with tab_elasticity:
    st.subheader("⚖️ Econometric Price Elasticity & Markdown Revenue Optimization")
    st.markdown(
        "Evaluate price responsiveness, demand inelasticity, and margin breakeven frontiers "
        "across catalog departments to prevent value-destructive discounts."
    )

    el_col1, el_col2 = st.columns([1, 2])
    with el_col1:
        selected_cat_el = st.selectbox(
            "Select Catalog Category",
            options=["Clothing", "Electronics", "Beauty"],
            index=1,
            key="el_cat_select",
        )
        cat_ped_res = elasticity_engine.compute_category_elasticity(selected_cat_el)

        st.metric("Log-Log PED Coefficient", f"{cat_ped_res['ped_coefficient']:+.4f}")
        st.metric("Regression R²", f"{cat_ped_res['r_squared']:.4f}")
        st.info(f"**Classification:** {cat_ped_res['classification']}")
        st.caption(cat_ped_res["interpretation"])

    with el_col2:
        st.markdown("#### 📉 Interactive Promotional Markdown Simulator")
        sim_disc = st.slider("Promotional Discount (%)", min_value=2, max_value=30, value=10, step=1) / 100.0
        sim_lift = st.slider("Expected Volume Lift (%)", min_value=-10, max_value=80, value=15, step=5) / 100.0

        promo_eval = elasticity_engine.simulate_markdown_pricing(
            category=selected_cat_el,
            discount_pct=sim_disc,
            expected_volume_lift_pct=sim_lift,
        )

        p_status = promo_eval["financial_impact"]["status"]
        if p_status == "Value Accretive":
            st.success(f"✅ **{p_status}**: Profit Delta {promo_eval['financial_impact']['profit_delta']:+,.2f} ({promo_eval['financial_impact']['profit_delta_pct']:+.1f}%)")
        elif "Dilutive" in p_status:
            st.warning(f"⚠️ **{p_status}**: Profit Delta {promo_eval['financial_impact']['profit_delta']:+,.2f} ({promo_eval['financial_impact']['profit_delta_pct']:+.1f}%)")
        else:
            st.error(f"❌ **{p_status}**: Revenue Delta {promo_eval['financial_impact']['revenue_delta']:+,.2f}")

        st.caption(
            f"**Breakeven Hurdle:** Requires at least **+{promo_eval['financial_impact']['breakeven_volume_lift_required_pct']:.1f}% volume lift** "
            f"to offset a {int(sim_disc*100)}% discount at 40% margin."
        )

        # Plotly comparison
        comp_df = pd.DataFrame([
            {"Stage": "Baseline", "Revenue": promo_eval["baseline"]["revenue"], "Gross Profit": promo_eval["baseline"]["gross_profit"]},
            {"Stage": "Post-Markdown", "Revenue": promo_eval["post_promotion"]["simulated_revenue"], "Gross Profit": promo_eval["post_promotion"]["simulated_gross_profit"]},
        ])
        fig_el = px.bar(
            comp_df,
            x="Stage",
            y=["Revenue", "Gross Profit"],
            barmode="group",
            text_auto="$,.0f",
            title=f"{selected_cat_el}: Baseline vs Markdown Financials",
            color_discrete_sequence=["#38bdf8", "#34d399"],
        )
        fig_el.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig_el, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🏷️ Discrete Catalog Price Tier Margin Architecture")
    tier_eval = elasticity_engine.optimize_price_points(selected_cat_el)
    st.dataframe(pd.DataFrame(tier_eval["tier_evaluations"]), use_container_width=True)


# ----------------- TAB 11: BAYESIAN A/B EXPERIMENTATION -----------------
with tab_ab_test:
    st.subheader("🧪 Bayesian & Frequentist Retail A/B Experimentation Suite")
    st.markdown(
        "Evaluate pilot retail promotions, bundle offers, and pricing experiments "
        "using rigorous dual hypothesis testing, Beta-Binomial conjugate updating, and power calculations."
    )

    ab_c1, ab_c2 = st.columns(2)
    with ab_c1:
        st.markdown("##### 🅰️ Control Group (Status Quo)")
        c_conv = st.number_input("Control Conversions / Adoptions", min_value=1, value=100, step=10)
        c_tot = st.number_input("Control Total Sample Size", min_value=10, value=1000, step=50)

    with ab_c2:
        st.markdown("##### 🅱️ Treatment Variant (New Promotion / Bundle)")
        t_conv = st.number_input("Variant Conversions / Adoptions", min_value=1, value=150, step=10)
        t_tot = st.number_input("Variant Total Sample Size", min_value=10, value=1000, step=50)

    # Run Bayesian & Frequentist tests
    ab_bayes = ab_engine.bayesian_conversion_test(
        control_conversions=c_conv,
        control_impressions=c_tot,
        treatment_conversions=t_conv,
        treatment_impressions=t_tot,
    )
    ab_freq = ab_engine.evaluate_proportion_ab(
        control_successes=c_conv,
        control_total=c_tot,
        treatment_successes=t_conv,
        treatment_total=t_tot,
    )

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Control Conversion", f"{(c_conv/c_tot)*100:.2f}%")
    with m_col2:
        st.metric("Variant Conversion", f"{(t_conv/t_tot)*100:.2f}%", f"{ab_freq['lift']['relative_lift_pct']:+.1f}% Lift")
    with m_col3:
        st.metric("P(Variant > Control)", f"{ab_bayes['bayesian_metrics']['prob_treatment_better_pct']:.1f}%")
    with m_col4:
        st.metric("Expected Loss", f"{ab_bayes['bayesian_metrics']['expected_loss_treatment']*100:.3f}%")

    b_dec = ab_bayes["recommendation"]["decision"]
    if "Deploy" in b_dec:
        st.success(f"🚀 **Decision: {b_dec}** — {ab_bayes['recommendation']['rationale']}")
    elif "Retain" in b_dec:
        st.error(f"🛑 **Decision: {b_dec}** — {ab_bayes['recommendation']['rationale']}")
    else:
        st.warning(f"⏳ **Decision: {b_dec}** — {ab_bayes['recommendation']['rationale']}")

    # Plot posterior curves
    x_grid = np.linspace(
        max(0, min(c_conv/c_tot, t_conv/t_tot) - 0.08),
        min(1.0, max(c_conv/c_tot, t_conv/t_tot) + 0.08),
        300,
    )
    y_c = stats.beta.pdf(x_grid, 1 + c_conv, 1 + c_tot - c_conv)
    y_t = stats.beta.pdf(x_grid, 1 + t_conv, 1 + t_tot - t_conv)

    fig_ab = go.Figure()
    fig_ab.add_trace(go.Scatter(x=x_grid*100, y=y_c, mode="lines", name="Control Posterior", line=dict(color="#94a3b8", width=2.5), fill="tozeroy"))
    fig_ab.add_trace(go.Scatter(x=x_grid*100, y=y_t, mode="lines", name="Variant Posterior", line=dict(color="#38bdf8", width=2.5), fill="tozeroy"))
    fig_ab.update_layout(
        title="Beta-Binomial Posterior Belief Distributions P(Conversion Rate | Data)",
        xaxis_title="Conversion Rate (%)",
        yaxis_title="Posterior Density",
        template="plotly_dark",
        height=340,
    )
    st.plotly_chart(fig_ab, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 📐 Sample Size Determination Matrix (Power = 80%, α = 0.05)")
    p_mat = ab_engine.generate_power_matrix(baseline_rate=c_conv/c_tot)
    st.dataframe(pd.DataFrame(p_mat), use_container_width=True)


# ----------------- TAB 12: CLV & CHURN EARLY WARNING -----------------
with tab_clv:
    st.subheader("💎 Customer Lifetime Value (CLV) & Churn Risk Early Warning Guard")
    st.markdown(
        "Monitor enterprise customer equity, survival probability decay, and high-monetary VIP whale churn risk "
        "to trigger high-ROI retention interventions."
    )

    clv_audit = clv_engine.audit_customer_equity()
    base_info = clv_audit["customer_base"]
    vip_info = clv_audit["vip_early_warning"]

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric("Total Projected Equity", f"${base_info['total_projected_12m_customer_equity']:,.0f}")
    with c_m2:
        st.metric("Average Customer CLV", f"${base_info['average_customer_clv']:,.2f}")
    with c_m3:
        st.metric("VIP Whales at Risk", f"{vip_info['vips_at_risk_count']} / {vip_info['total_vips']}")
    with c_m4:
        st.metric("VIP Equity at Risk", f"${vip_info['vip_revenue_at_risk']:,.0f}")

    clv_col1, clv_col2 = st.columns([1, 1])
    with clv_col1:
        st.markdown("##### 👥 Customer Equity by Lifecycle Risk Tier")
        t_df = pd.DataFrame(clv_audit["churn_tier_distribution"])
        fig_tiers = px.pie(
            t_df,
            names="churn_tier",
            values="total_projected_12m_clv",
            title="12-Month Projected Customer Equity Distribution",
            color="churn_tier",
            color_discrete_map={
                "Active (Low Risk)": "#22c55e",
                "Cooling (Medium Risk)": "#eab308",
                "At Risk (High Risk)": "#f97316",
                "Lapsed (Defected)": "#ef4444",
            },
            hole=0.45,
        )
        fig_tiers.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig_tiers, use_container_width=True)

    with clv_col2:
        st.markdown("##### 🔮 Interactive Individual Customer CLV Predictor")
        in_rec = st.slider("Customer Recency (Days Elapsed Since Order)", min_value=1, max_value=365, value=45, step=5)
        in_aov = st.slider("Historical Basket Spend ($)", min_value=25, max_value=2000, value=500, step=25)

        ind_pred = clv_engine.predict_customer_clv(recency_days=in_rec, aov=in_aov)
        st.info(
            f"**Risk Tier:** `{ind_pred['churn_tier']}` · **Churn Probability:** `{ind_pred['prob_churn']*100:.1f}%`\n\n"
            f"**Projected 12m CLV:** `${ind_pred['projected_clv']:,.2f}`\n\n"
            f"**Action Playbook:** {ind_pred['recommended_playbook']}"
        )

    st.markdown("---")
    st.markdown("##### 🚨 High-Priority At-Risk VIP Whales (Immediate Intervention Target)")
    at_risk_list = clv_engine.get_at_risk_vips(limit=10)
    st.dataframe(pd.DataFrame(at_risk_list), use_container_width=True)


# ----------------- TAB 13: SUPPLY CHAIN & INVENTORY OPTIMIZATION -----------------
with tab_inventory:
    st.subheader("📦 Supply Chain & Inventory Optimization (EOQ, Dynamic ROP & Safety Stock)")
    st.markdown(
        "Calibrate safety stock buffers, dynamic reorder points, economic order batch sizes, "
        "and inventory working capital requirements under lead time uncertainty."
    )

    inv_summary = inventory_engine.get_storewide_inventory_summary(service_level=0.95)
    i_col1, i_col2, i_col3, i_col4 = st.columns(4)
    with i_col1:
        st.metric("Total Working Capital Committed", f"${inv_summary['total_working_capital_committed']:,.2f}")
    with i_col2:
        st.metric("Safety Buffer Capital", f"${inv_summary['total_safety_stock_capital']:,.2f}")
    with i_col3:
        st.metric("Cycle Stock Capital", f"${inv_summary['total_cycle_stock_capital']:,.2f}")
    with i_col4:
        st.metric("Annual Inventory Holding Cost", f"${inv_summary['total_annual_holding_cost']:,.2f}")

    st.markdown("---")
    sc_col1, sc_col2 = st.columns([1, 2])
    with sc_col1:
        st.markdown("##### ⚙️ Replenishment Policy Calibrator")
        target_sl = st.slider("Target Cycle Service Level (SL)", min_value=0.80, max_value=0.999, value=0.95, step=0.005)
        lead_time = st.number_input("Supplier Lead Time (Days)", min_value=1.0, max_value=60.0, value=7.0, step=1.0)
        lead_std = st.number_input("Lead Time Std Dev (Days)", min_value=0.0, max_value=10.0, value=1.5, step=0.5)
        sel_cat = st.selectbox("Merchandise Category", ["All Categories", "Clothing", "Electronics", "Beauty"])
        query_cat = None if sel_cat == "All Categories" else sel_cat

        cat_ss = inventory_engine.calculate_safety_stock_and_rop(
            service_level=target_sl, lead_time_days=lead_time, lead_time_std=lead_std, category=query_cat
        )
        cat_eoq = inventory_engine.calculate_eoq(category=query_cat)
        cat_risk = inventory_engine.simulate_stockout_risk(
            service_level=target_sl, lead_time_days=lead_time, lead_time_std=lead_std, category=query_cat
        )

        st.info(
            f"**Safety Stock (SS):** `{cat_ss['safety_stock_units']} units` (${cat_ss['safety_stock_capital']:,.2f})\n\n"
            f"**Reorder Point (ROP):** `{cat_ss['reorder_point_units']} units`\n\n"
            f"**Economic Batch Size (EOQ):** `{cat_eoq['eoq_units']} units`\n\n"
            f"**Replenishment Interval:** Every `{cat_eoq['order_cycle_interval_days']} days`\n\n"
            f"**Cycle Fill Rate:** `{cat_risk['cycle_fill_rate_percentage']:.2f}%` · **Lost Margin:** `${cat_risk['annual_lost_gross_margin']:,.2f}/yr`"
        )

    with sc_col2:
        st.markdown("##### 📈 Service Level vs. Working Capital Investment Sensitivity")
        sl_range = np.linspace(0.80, 0.999, 40)
        ss_capitals = []
        for sl_val in sl_range:
            res_val = inventory_engine.calculate_safety_stock_and_rop(
                service_level=float(sl_val), lead_time_days=lead_time, lead_time_std=lead_std, category=query_cat
            )
            ss_capitals.append(res_val["safety_stock_capital"])

        fig_inv = go.Figure()
        fig_inv.add_trace(go.Scatter(x=sl_range * 100.0, y=ss_capitals, mode="lines+markers", name="Safety Stock Capital ($)", line=dict(color="#38bdf8", width=3)))
        fig_inv.update_layout(
            title=f"Non-Linear Capital Escalation Curve ({sel_cat})",
            xaxis_title="Target Cycle Service Level (%)",
            yaxis_title="Safety Stock Capital Commitment ($)",
            template="plotly_dark",
            height=340,
        )
        st.plotly_chart(fig_inv, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 📊 Multi-Criteria ABC-FSN Inventory Stratification Matrix")
    abc_fsn = inventory_engine.perform_abc_fsn_analysis()
    st.dataframe(pd.DataFrame(abc_fsn["sku_stratifications"]), use_container_width=True)


# ----------------- TAB 14: MARKET BASKET & CROSS-MERCHANDISING -----------------
with tab_market_basket:
    st.subheader("🛒 Market Basket Association Rules & Cross-Merchandising Synergy Miner")
    st.markdown(
        "Uncover directional category co-occurrence, association lift, and margin-protected "
        "promotional bundling opportunities across merchandise departments."
    )

    affinity_data = market_basket_engine.build_cooccurrence_and_affinity_matrix()
    mb_col1, mb_col2 = st.columns([1, 1])

    with mb_col1:
        st.markdown("##### 🔗 Directional Cross-Category Lift & Synergy Heatmap")
        rules_df = pd.DataFrame(affinity_data["rules"])
        pivot_lift = rules_df.pivot(index="antecedent", columns="consequent", values="lift")

        fig_heat = px.imshow(
            pivot_lift,
            text_auto=".2f",
            color_continuous_scale="Viridis",
            title="Association Lift (Values > 1.0 Indicate Complementary Cross-Selling)",
            labels=dict(x="Consequent (Add-On)", y="Antecedent (Base Category)", color="Lift"),
        )
        fig_heat.update_layout(template="plotly_dark", height=340)
        st.plotly_chart(fig_heat, use_container_width=True)

    with mb_col2:
        st.markdown("##### 🎁 Interactive Cross-Department Bundle Recommender")
        t_cat = st.selectbox("Select Base Merchandise Department", ["Electronics", "Clothing", "Beauty"])
        disc_pct = st.slider("Promotional Bundle Discount (%)", min_value=5, max_value=25, value=12, step=1)

        combos = market_basket_engine.recommend_merchandising_combos(
            target_category=t_cat, discount_rate=disc_pct / 100.0
        )
        for combo in combos:
            st.info(
                f"**Bundle Pairing:** `{combo['base_category']} + {combo['complementary_category']}`\n\n"
                f"**Lift:** `{combo['association_lift']:.2f}` · **Confidence:** `{combo['purchase_confidence']*100:.1f}%`\n\n"
                f"**Retail Pricing:** ~~${combo['original_bundle_price']:.2f}~~ ➔ **${combo['discounted_bundle_price']:.2f}** (Save ${combo['customer_savings']:.2f})\n\n"
                f"**Retained Margin:** `{combo['estimated_retained_margin_pct']:.1f}%` · **Synergy Score:** `{combo['synergy_score']:.1f}`"
            )

    st.markdown("---")
    st.markdown("##### 👥 Demographic Age Cohort Category Affinity Breakdown")
    demo_breakdown = market_basket_engine.demographic_affinity_breakdown()
    demo_records = []
    for cohort_name, c_info in demo_breakdown["cohort_breakdown"].items():
        demo_records.append({
            "Age Cohort": cohort_name,
            "Total Transactions": c_info["total_transactions"],
            "Dominant Category": c_info["dominant_category"],
            "Avg Basket Spend ($)": c_info["average_basket_spend"],
            "Multi-Category Affinity Index": c_info["multi_category_affinity_index"],
        })
    st.dataframe(pd.DataFrame(demo_records), use_container_width=True)


# ----------------- TAB 15: MULTI-HORIZON DEMAND FORECASTING -----------------
with tab_forecasting:
    st.subheader("⏱️ Multi-Horizon Demand Forecasting & Conformal Uncertainty Bounds")
    st.markdown(
        "Project future unit demand and revenue with distribution-free conformal prediction "
        "intervals, walk-forward validation accuracy, and rolling anomaly detection."
    )

    f_acc = forecasting_engine.evaluate_forecast_accuracy(test_window_days=45)
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.metric("Holdout MAE", f"{f_acc['mae_units']:.2f} units")
    with fc2:
        st.metric("Holdout RMSE", f"{f_acc['rmse_units']:.2f} units")
    with fc3:
        st.metric("Weighted APE (WAPE)", f"{f_acc['wape_percentage']:.2f}%")
    with fc4:
        st.metric("Conformal 95% Coverage", f"{f_acc['conformal_95_coverage_percentage']:.1f}%")

    st.markdown("---")
    f_cat_sel = st.selectbox("Select Forecast Category", ["All Categories", "Clothing", "Electronics", "Beauty"], key="fc_cat_sel")
    q_f_cat = None if f_cat_sel == "All Categories" else f_cat_sel
    f_horizon = st.slider("Select Forecast Horizon (Days Ahead)", min_value=7, max_value=60, value=30, step=7)

    forecast_res = forecasting_engine.forecast_demand(horizon_days=f_horizon, category=q_f_cat)
    daily_hist = forecasting_engine.get_daily_demand_series(category=q_f_cat)

    pred_df = pd.DataFrame(forecast_res["daily_projections"])
    pred_df["Date"] = pd.to_datetime(pred_df["date"])

    fig_fc = go.Figure()
    hist_slice = daily_hist.iloc[-90:]
    fig_fc.add_trace(go.Scatter(x=hist_slice["Date"], y=hist_slice["units"], mode="lines", name="Historical Daily Demand", line=dict(color="#94a3b8", width=1.5)))
    fig_fc.add_trace(go.Scatter(x=hist_slice["Date"], y=hist_slice["rolling_7d_mean"], mode="lines", name="Historical 7D Velocity", line=dict(color="#f59e0b", width=2)))

    fig_fc.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["predicted_units"], mode="lines+markers", name="Point Forecast", line=dict(color="#38bdf8", width=2.5)))

    fig_fc.add_trace(go.Scatter(
        x=pd.concat([pred_df["Date"], pred_df["Date"][::-1]]),
        y=pd.concat([pred_df["upper_bound_95"], pred_df["lower_bound_95"][::-1]]),
        fill="toself",
        fillcolor="rgba(56, 189, 248, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Conformal Prediction Interval",
    ))

    fig_fc.update_layout(
        title=f"Multi-Horizon Forward Projection & Conformal Uncertainty: {f_cat_sel} ({f_horizon} Days)",
        xaxis_title="Date",
        yaxis_title="Units Demanded",
        template="plotly_dark",
        height=380,
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("##### 🚨 Historical Demand Surge & Volatility Spikes (Rolling Z-Score Anomaly Alerts)")
    anomalies = forecasting_engine.detect_demand_anomalies(category=q_f_cat, z_threshold=2.2)
    if anomalies:
        st.dataframe(pd.DataFrame(anomalies).head(10), use_container_width=True)
    else:
        st.success("No critical demand anomalies detected at threshold Z >= 2.2.")



# ----------------- TAB 16: MARKETING ATTRIBUTION & ROAS -----------------
with tab_attribution:
    st.subheader("🎯 Omnichannel Multi-Touch Attribution & Marketing ROAS Optimization")
    st.markdown(
        "Evaluate marketing channel contribution across 5 heuristic models, First-Order Markov removal effects, "
        "and execute Markowitz Modern Portfolio Theory (MPT) optimal advertising budget allocation."
    )

    attr_heuristics = attribution_engine.get_heuristic_attribution_summary()
    attr_markov = attribution_engine.compute_markov_attribution()

    col_att1, col_att2, col_att3, col_att4 = st.columns(4)
    with col_att1:
        st.metric("Total Conversion Events", f"{attr_heuristics['total_transactions']:,}")
    with col_att2:
        st.metric("Attributed Revenue", f"${attr_heuristics['total_revenue']:,.2f}")
    with col_att3:
        st.metric("Markov Baseline Conversion", f"{attr_markov['baseline_conversion_probability']*100:.2f}%")
    with col_att4:
        st.metric("Evaluated Channels", f"{attr_markov['channels_evaluated']}")

    st.markdown("---")
    st.markdown("##### 📊 Cross-Model Attributed Revenue Comparison ($)")
    sel_model = st.selectbox(
        "Select Attribution Model to Inspect",
        ["First-Touch", "Last-Touch", "Linear", "Time-Decay", "Position-Based"],
        index=3,
        key="sel_attr_model",
    )

    model_channels = attr_heuristics["models"][sel_model]
    df_model = pd.DataFrame(model_channels)

    fig_attr = px.bar(
        df_model,
        x="channel",
        y="attributed_revenue",
        color="roas",
        color_continuous_scale="Viridis",
        title=f"Attributed Revenue & ROAS by Channel: {sel_model}",
        labels={"attributed_revenue": "Attributed Revenue ($)", "channel": "Marketing Channel", "roas": "ROAS (x)"},
        text_auto=".2s",
        template="plotly_dark",
    )
    st.plotly_chart(fig_attr, use_container_width=True)

    st.markdown("##### 🔬 First-Order Markov Chain Removal Effects & Efficiency Frontier")
    df_markov = pd.DataFrame(attr_markov["channel_attribution"])
    st.dataframe(df_markov, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 💼 Markowitz Portfolio Optimal Marketing Budget Allocation")
    col_bg1, col_bg2 = st.columns(2)
    with col_bg1:
        slider_budget = st.slider("Total Advertising Budget ($)", min_value=30000.0, max_value=150000.0, value=80000.0, step=5000.0)
    with col_bg2:
        slider_risk = st.slider("Risk Aversion Parameter (Gamma)", min_value=0.0, max_value=1.5, value=0.5, step=0.1)

    opt_budget_res = attribution_engine.optimize_marketing_budget(total_budget=slider_budget, risk_aversion=slider_risk)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Baseline Revenue", f"${opt_budget_res['baseline_revenue']:,.2f}")
    with m2:
        st.metric("Optimized Projected Revenue", f"${opt_budget_res['projected_optimal_revenue']:,.2f}")
    with m3:
        st.metric("Projected Revenue Uplift", f"+${opt_budget_res['projected_revenue_uplift']:,.2f}", f"+{opt_budget_res['uplift_pct']:.1f}%")
    with m4:
        st.metric("Projected Blended ROAS", f"{opt_budget_res['projected_blended_roas']:.2f}x", f"vs {opt_budget_res['baseline_blended_roas']:.2f}x")

    st.dataframe(pd.DataFrame(opt_budget_res["channel_allocations"]), use_container_width=True)


# ----------------- TAB 17: GMROI & ASSORTMENT RATIONALIZER -----------------
with tab_gmroi:
    st.subheader("💎 Gross Margin Return on Investment (GMROI) & Merchandise Assortment")
    st.markdown(
        "Evaluate inventory capital productivity, diagnose the Margin-Turn 4-Quadrant Matrix, "
        "and simulate capital reallocation from dead stock to high-performing Stars."
    )

    gmroi_cat = gmroi_engine.calculate_category_gmroi()
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.metric("Store Total Gross Profit", f"${gmroi_cat['store_total_gross_profit']:,.2f}")
    with g2:
        st.metric("Store Blended Gross Margin", f"{gmroi_cat['store_blended_gross_margin_pct']:.1f}%")
    with g3:
        st.metric("Total Inventory Capital Tied Up", f"${gmroi_cat['store_total_inventory_investment']:,.2f}")
    with g4:
        st.metric("Store Blended GMROI", f"{gmroi_cat['store_blended_gmroi']:.2f}x")

    st.markdown("---")
    st.markdown("##### 📊 Margin-Turn Strategic 4-Quadrant Matrix")
    matrix_data = gmroi_engine.compute_margin_turn_matrix()
    sku_df = pd.DataFrame(matrix_data["classified_skus"])

    fig_matrix = px.scatter(
        sku_df,
        x="inventory_turns",
        y="gross_margin_pct",
        size="revenue",
        color="quadrant",
        hover_data=["sku_id", "category", "price_tier", "gmroi", "merchandising_action"],
        title="Merchandise Portfolio Matrix: Turns vs Gross Margin % (Bubble Size = Annual Revenue)",
        labels={"inventory_turns": "Annual Inventory Turns", "gross_margin_pct": "Gross Margin %", "quadrant": "Strategic Quadrant"},
        color_discrete_map={
            "Stars": "#22c55e",
            "High-Yield Specialists": "#38bdf8",
            "Traffic Drivers": "#f59e0b",
            "Dead Stock / Capital Traps": "#ef4444",
        },
        template="plotly_dark",
    )
    fig_matrix.add_vline(x=matrix_data["thresholds"]["median_inventory_turns"], line_dash="dash", line_color="gray")
    fig_matrix.add_hline(y=matrix_data["thresholds"]["median_gross_margin_pct"], line_dash="dash", line_color="gray")
    st.plotly_chart(fig_matrix, use_container_width=True)

    st.markdown("##### 🚨 SKU Assortment Rationalization Candidates (Capital Drag Diagnostics)")
    diag_data = gmroi_engine.evaluate_assortment_rationalization()
    st.dataframe(pd.DataFrame(diag_data["candidates"]), use_container_width=True)

    st.markdown("---")
    st.markdown("##### 🔄 Working Capital Reallocation Simulator")
    realloc_pct = st.slider("Reallocate Inventory Capital from Dead Stock to Stars (%)", min_value=10, max_value=70, value=35, step=5)
    realloc_sim = gmroi_engine.simulate_working_capital_reallocation(reallocation_pct=realloc_pct / 100.0)

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.metric("Capital Reallocated", f"${realloc_sim['capital_reallocated_dollars']:,.2f}")
    with rc2:
        st.metric("Incremental Gross Profit", f"+${realloc_sim['incremental_gross_profit']:,.2f}", f"+{realloc_sim['profit_uplift_pct']:.1f}%")
    with rc3:
        st.metric("Projected Store GMROI", f"{realloc_sim['projected_store_gmroi']:.2f}x", f"+{realloc_sim['gmroi_delta']:.2f}x")


# ----------------- TAB 18: COHORT RETENTION & SURVIVAL -----------------
with tab_cohorts:
    st.subheader("⏳ Customer Cohort Retention Dynamics & Non-Parametric Survival Analysis")
    st.markdown(
        "Track monthly customer acquisition cohort decay, estimate Kaplan-Meier survival curves S(t), "
        "execute demographic log-rank hypothesis tests, and pinpoint acute churn cliff horizons."
    )

    cohort_res = cohort_engine.compute_cohort_retention_matrix()
    km_res = cohort_engine.compute_kaplan_meier_survival(segment_by="gender")
    hazard_res = cohort_engine.detect_churn_hazard_cliffs(interval_days=20)

    ch1, ch2, ch3, ch4 = st.columns(4)
    with ch1:
        st.metric("Total Monitored Cohorts", f"{cohort_res['total_cohorts']}")
    with ch2:
        st.metric("Median Customer Survival", f"{km_res['overall_survival']['median_survival_days']} days")
    with ch3:
        st.metric("Median Churn Hazard Rate", f"{hazard_res['median_hazard_rate']:.4f}")
    with ch4:
        st.metric("Acute Churn Cliffs Detected", f"{hazard_res['critical_churn_cliffs_detected']}")

    st.markdown("---")
    st.markdown("##### 📈 Kaplan-Meier Survival Functions S(t) with Greenwood 95% Confidence Band")

    km_overall = pd.DataFrame(km_res["overall_survival"]["timeline"])
    fig_km = go.Figure()
    fig_km.add_trace(go.Scatter(
        x=km_overall["day"],
        y=km_overall["survival_probability"],
        mode="lines",
        name="Overall Kaplan-Meier S(t)",
        line=dict(color="#38bdf8", width=3),
    ))
    fig_km.add_trace(go.Scatter(
        x=pd.concat([km_overall["day"], km_overall["day"][::-1]]),
        y=pd.concat([km_overall["ci_upper_95"], km_overall["ci_lower_95"][::-1]]),
        fill="toself",
        fillcolor="rgba(56, 189, 248, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Greenwood Confidence Band",
    ))
    fig_km.update_layout(
        title="Kaplan-Meier Customer Survival Function S(t)",
        xaxis_title="Tenure (Days Since Acquisition)",
        yaxis_title="Estimated Survival Probability",
        template="plotly_dark",
        height=380,
    )
    st.plotly_chart(fig_km, use_container_width=True)

    st.markdown("##### ⚖️ Demographic Log-Rank Hypothesis Test")
    sel_seg = st.selectbox("Select Segment to Compare", ["gender", "category"], key="sel_logrank_seg")
    lr_res = cohort_engine.compute_log_rank_test(segment_col=sel_seg)
    if "error" not in lr_res:
        st.info(f"**Log-Rank Comparison:** `{lr_res['comparison']}` | **Chi-Square Stat:** `{lr_res['chi_square_statistic']}` | **p-value:** `{lr_res['p_value']}` | **Verdict:** {lr_res['conclusion']}")

    st.markdown("---")
    st.markdown("##### 💰 Proactive Retention Campaign ROI Calculator")
    col_ret1, col_ret2 = st.columns(2)
    with col_ret1:
        cost_per_touch = st.number_input("Intervention Cost per Customer ($)", min_value=1.0, max_value=100.0, value=12.0, step=1.0)
    with col_ret2:
        churn_reduc_pct = st.slider("Expected Churn Reduction Rate (%)", min_value=5, max_value=60, value=25, step=5)

    ret_roi = cohort_engine.calculate_retention_campaign_roi(
        intervention_cost_per_user=cost_per_touch,
        expected_churn_reduction_pct=churn_reduc_pct / 100.0,
    )

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Target Population", f"{ret_roi['target_population']:,}")
    with r2:
        st.metric("Total Campaign Cost", f"${ret_roi['total_campaign_cost']:,.2f}")
    with r3:
        st.metric("Projected Saved Shoppers", f"{ret_roi['projected_saved_customers']:.1f}")
    with r4:
        st.metric("Net Commercial Gain", f"+${ret_roi['net_commercial_gain']:,.2f}", f"{ret_roi['campaign_roi_pct']:.1f}% ROI")


# ----------------- TAB 19: DYNAMIC BANDIT PRICING -----------------
with tab_dynamic_pricing:
    st.subheader("🎯 Real-Time Dynamic Pricing & Multi-Armed Bandit (MAB) Rate Optimizer")
    st.markdown(
        "Explore real-time dynamic pricing quotes driven by inventory scarcity multipliers, "
        "demand velocity surges, and evaluate Thompson Sampling vs UCB1 regret minimization."
    )

    arms_catalog = dynamic_pricing_engine.get_candidate_arms()
    dp1, dp2, dp3 = st.columns(3)
    with dp1:
        st.metric("Supported MAB Algorithms", f"{len(arms_catalog['algorithms_supported'])} Algorithms")
    with dp2:
        st.metric("Active Catalog Price Arms", "15 Discrete Arms")
    with dp3:
        st.metric("Guardrails Protection", "Margin Floors & Ceilings Enforced")

    st.markdown("---")
    st.markdown("##### ⚡ Real-Time Dynamic Pricing Quote Sandbox")
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        sel_cat = st.selectbox("Merchandise Category", ["Beauty", "Clothing", "Electronics"], key="sel_quote_cat")
    with qc2:
        inp_base_price = st.number_input("Catalog Base List Price ($)", min_value=10.0, max_value=800.0, value=300.0 if sel_cat == "Electronics" else (50.0 if sel_cat == "Clothing" else 40.0), step=5.0)
    with qc3:
        inp_stock = st.number_input("Current On-Hand Stock (Units)", min_value=0, max_value=150, value=12, step=1)
    with qc4:
        inp_velocity = st.slider("7-Day Sales Velocity (vs Baseline)", min_value=0.5, max_value=2.0, value=1.35, step=0.05)

    quote_res = dynamic_pricing_engine.calculate_dynamic_quote(
        category=sel_cat,
        base_price=inp_base_price,
        current_stock=inp_stock,
        safety_stock=25,
        demand_velocity_ratio=inp_velocity,
    )

    qm1, qm2, qm3, qm4 = st.columns(4)
    with qm1:
        st.metric("Recommended Dynamic Price", f"${quote_res['recommended_dynamic_price']:,.2f}", f"{quote_res['price_delta_pct']:+.1f}% vs List")
    with qm2:
        st.metric("Scarcity Multiplier", f"{quote_res['multipliers']['scarcity_multiplier']:.2f}x", f"Stock: {quote_res['stock_context']['stock_status']}")
    with qm3:
        st.metric("Demand Surge Multiplier", f"{quote_res['multipliers']['demand_surge_multiplier']:.2f}x")
    with qm4:
        st.metric("Projected Gross Margin", f"{quote_res['economics']['projected_gross_margin_pct']:.1f}%", f"${quote_res['economics']['projected_gross_margin_dollars']:,.2f}/unit")

    st.info(f"**Pricing Action Strategy:** `{quote_res['pricing_action']}` | **Price Floor:** `${quote_res['guardrails']['price_floor']:.2f}` | **Price Ceiling:** `${quote_res['guardrails']['price_ceiling']:.2f}`")

    st.markdown("---")
    st.markdown("##### 🎰 Multi-Armed Bandit Online Exploration & Regret Minimization Simulation")
    bc1, bc2 = st.columns(2)
    with bc1:
        sel_algo = st.selectbox("Bandit Learning Algorithm", ["thompson_sampling", "ucb1", "epsilon_greedy"], key="sel_mab_algo")
    with bc2:
        n_sim_rounds = st.slider("Customer Shopping Impression Rounds", min_value=100, max_value=2000, value=800, step=100)

    sim_mab = dynamic_pricing_engine.simulate_bandit_learning(
        category=sel_cat,
        algorithm=sel_algo,
        n_rounds=n_sim_rounds,
        epsilon=0.10,
    )

    bm1, bm2, bm3, bm4 = st.columns(4)
    with bm1:
        st.metric("Bandit Realized Revenue", f"${sim_mab['cumulative_bandit_revenue']:,.2f}")
    with bm2:
        st.metric("Static Pricing Baseline", f"${sim_mab['cumulative_static_revenue']:,.2f}")
    with bm3:
        st.metric("Dynamic Revenue Uplift", f"+${sim_mab['revenue_uplift_vs_static']:,.2f}", f"+{sim_mab['revenue_uplift_pct']:.1f}%")
    with bm4:
        st.metric("Cumulative Regret vs Oracle", f"${sim_mab['total_regret']:,.2f}", f"${sim_mab['regret_per_round']:.2f}/round")

    # Plot revenue comparison and regret
    df_curve = pd.DataFrame(sim_mab["learning_curve"])
    fig_bandit = go.Figure()
    fig_bandit.add_trace(go.Scatter(x=df_curve["rounds"], y=df_curve["bandit_revenue"], name=f"Bandit ({sel_algo})", line=dict(color="#22c55e", width=3)))
    fig_bandit.add_trace(go.Scatter(x=df_curve["rounds"], y=df_curve["static_revenue"], name="Static List Pricing", line=dict(color="#f59e0b", dash="dash", width=2)))
    fig_bandit.add_trace(go.Scatter(x=df_curve["rounds"], y=df_curve["oracle_revenue"], name="Oracle Optimal Ceiling", line=dict(color="#38bdf8", dash="dot", width=2)))
    fig_bandit.update_layout(title=f"Cumulative Revenue Trajectory: Bandit Learning vs Static ({sel_cat})", xaxis_title="Simulation Round", yaxis_title="Cumulative Revenue ($)", template="plotly_dark", height=380)
    st.plotly_chart(fig_bandit, use_container_width=True)


# ----------------- TAB 20: CAUSAL UPLIFT & NEXT-BEST-ACTION -----------------
with tab_nba:
    st.subheader("🎯 Causal Uplift Micro-Targeting & Next-Best-Action (NBA) Decision Engine")
    st.markdown(
        "Segment customers into the 4 causal response quadrants to target Persuadables, "
        "suppress Sure Things to prevent margin cannibalization, and recommend personalized Next-Best-Actions."
    )

    uplift_summary = nba_engine.get_uplift_summary()
    u1, u2, u3, u4 = st.columns(4)
    with u1:
        st.metric("Customer Population", f"{uplift_summary['total_customers']:,}")
    with u2:
        st.metric("Persuadables (Prime Target)", f"{uplift_summary['persuadable_population']:,}", f"{uplift_summary['persuadable_share_pct']:.1f}% Share")
    with u3:
        st.metric("Mean Population Uplift (Tau)", f"+{uplift_summary['overall_mean_uplift_tau']:.4f}")
    with u4:
        st.metric("Cannibalization Safeguard", "Active (Sure Things Suppressed)")

    st.markdown("---")
    st.markdown("##### 🧬 Causal Uplift 4-Quadrant Population Breakdown")
    df_quad = pd.DataFrame(uplift_summary["quadrant_breakdown"])
    fig_quad = px.bar(
        df_quad,
        x="quadrant",
        y="customer_count",
        color="quadrant",
        color_discrete_map={
            "Persuadables": "#22c55e",
            "Sure Things": "#38bdf8",
            "Lost Causes": "#94a3b8",
            "Sleeping Dogs / Do Not Disturb": "#ef4444",
        },
        title="Customer Distribution by Causal Treatment Response Quadrant",
        labels={"customer_count": "Customers", "quadrant": "Causal Quadrant"},
        text_auto=True,
        template="plotly_dark",
    )
    st.plotly_chart(fig_quad, use_container_width=True)
    st.dataframe(df_quad[["quadrant", "customer_count", "population_share_pct", "avg_historical_spend", "avg_incremental_uplift_tau", "strategic_recommendation"]], use_container_width=True)

    st.markdown("---")
    st.markdown("##### 🤖 Personalized Next-Best-Action (NBA) Ranking Engine")
    nbc1, nbc2, nbc3 = st.columns(3)
    with nbc1:
        inp_cust_id = st.text_input("Customer ID (e.g. CUST001 to CUST999)", value="CUST001")
    with nbc2:
        inp_aov = st.number_input("Expected Order Basket ($)", min_value=50.0, max_value=1500.0, value=300.0, step=25.0)
    with nbc3:
        inp_recent_touches = st.slider("Recent Marketing Contacts (Fatigue Count)", min_value=0, max_value=5, value=1)

    nba_rec = nba_engine.recommend_next_best_action(
        customer_id=inp_cust_id if inp_cust_id else None,
        estimated_aov=inp_aov,
        gross_margin_pct=0.45,
        recent_touches_count=inp_recent_touches,
    )

    st.success(f"**Top Recommended Action:** `{nba_rec['recommended_action']}` via **Channel:** `{nba_rec['recommended_channel']}` | **Net Utility:** `+${nba_rec['projected_net_utility']:.2f}`")
    st.dataframe(pd.DataFrame(nba_rec["action_portfolio_ranking"]), use_container_width=True)

    st.markdown("---")
    st.markdown("##### 💰 Campaign Targeting Benchmark: Mass vs Propensity vs Causal Uplift")
    opt_camp = nba_engine.optimize_campaign_budget(campaign_budget=10000.0, incentive_cost_per_touch=8.0, avg_basket_value=350.0, gross_margin_pct=0.45)
    st.dataframe(pd.DataFrame(list(opt_camp["strategies"].values())), use_container_width=True)


# ----------------- TAB 21: WORKING CAPITAL & STRESS TEST -----------------
with tab_finance:
    st.subheader("🏦 Retail Working Capital, Cash Conversion Cycle & Monte Carlo Stress Testing")
    st.markdown(
        "Evaluate the enterprise Cash Conversion Cycle (DIO + DSO - DPO) across categories, "
        "and simulate 10,000 forward-looking stochastic macroeconomic stress scenarios calculating VaR 95/99 and CVaR."
    )

    ccc_data = financial_engine.calculate_cash_conversion_cycle()
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.metric("Store Blended CCC", f"{ccc_data['store_blended_ccc_days']:.1f} Days")
    with fc2:
        st.metric("Days Inventory Outstanding (DIO)", f"{ccc_data['store_blended_dio']:.1f} Days")
    with fc3:
        st.metric("Days Sales Outstanding (DSO)", f"{ccc_data['store_blended_dso']:.1f} Days")
    with fc4:
        st.metric("Days Payables Outstanding (DPO)", f"{ccc_data['store_blended_dpo']:.1f} Days")

    st.markdown("---")
    st.markdown("##### 📊 Category Working Capital & Cash Conversion Cycle Breakdown")
    df_ccc_cat = pd.DataFrame(ccc_data["category_metrics"])
    st.dataframe(df_ccc_cat[["category", "annual_revenue", "annual_cogs", "days_inventory_outstanding_dio", "days_sales_outstanding_dso", "days_payables_outstanding_dpo", "cash_conversion_cycle_days", "net_working_capital_required"]], use_container_width=True)

    st.markdown("---")
    st.markdown("##### 🎲 Monte Carlo Forward-Looking Macroeconomic Liquidity Stress Test")
    col_sc1, col_sc2, col_sc3 = st.columns(3)
    with col_sc1:
        inp_demand_shock = st.slider("Mean Consumer Demand Contraction (%)", min_value=-30.0, max_value=0.0, value=-10.0, step=1.0)
    with col_sc2:
        inp_cogs_infl = st.slider("Mean Supplier COGS Inflation (%)", min_value=0.0, max_value=20.0, value=6.0, step=1.0)
    with col_sc3:
        inp_cash_buf = st.number_input("Starting Treasury Cash Buffer ($)", min_value=10000.0, max_value=200000.0, value=60000.0, step=5000.0)

    stress_res = financial_engine.run_monte_carlo_stress_test(
        n_simulations=2000,
        demand_shock_mean_pct=inp_demand_shock / 100.0,
        demand_shock_std_pct=0.06,
        cogs_inflation_mean_pct=inp_cogs_infl / 100.0,
        cogs_inflation_std_pct=0.03,
        dpo_contraction_days=5.0,
        initial_cash_buffer=inp_cash_buf,
    )

    sm1, sm2, sm3, sm4 = st.columns(4)
    with sm1:
        st.metric("Value at Risk (VaR 95%)", f"${stress_res['stressed_outcomes']['var_95_cash_flow']:,.2f}")
    with sm2:
        st.metric("Extreme Tail (VaR 99%)", f"${stress_res['stressed_outcomes']['var_99_cash_flow']:,.2f}")
    with sm3:
        st.metric("Expected Shortfall (CVaR 95%)", f"${stress_res['stressed_outcomes']['cvar_95_expected_shortfall']:,.2f}")
    with sm4:
        st.metric("Insolvency Depletion Risk", f"{stress_res['liquidity_and_solvency']['insolvency_risk_pct']:.1f}%", f"Runway: {stress_res['liquidity_and_solvency']['average_solvency_runway_months']:.1f} mos")

    # Plot cash flow distribution
    df_dist = pd.DataFrame(stress_res["cash_flow_distribution"])
    fig_stress = px.bar(
        df_dist,
        x="bin_start",
        y="frequency",
        title="Stressed Annual Operating Cash Flow Distribution (Monte Carlo 2,000 Paths)",
        labels={"bin_start": "Operating Cash Flow ($)", "frequency": "Scenario Count"},
        template="plotly_dark",
    )
    fig_stress.add_vline(x=stress_res['stressed_outcomes']['var_95_cash_flow'], line_dash="dash", line_color="#f59e0b", annotation_text="VaR 95%")
    fig_stress.add_vline(x=stress_res['stressed_outcomes']['var_99_cash_flow'], line_dash="dash", line_color="#ef4444", annotation_text="VaR 99%")
    st.plotly_chart(fig_stress, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 🔄 Working Capital Optimization Simulator")
    rebal_sim = financial_engine.simulate_working_capital_optimization(dio_reduction_days=12.0, dpo_extension_days=10.0, dso_reduction_days=1.0)
    w1, w2, w3 = st.columns(3)
    with w1:
        st.metric("Permanent Liquidity Unlocked", f"${rebal_sim['optimized_outcomes']['total_permanent_liquidity_unlocked']:,.2f}")
    with w2:
        st.metric("CCC Days Eliminated", f"-{rebal_sim['optimized_outcomes']['ccc_reduction_days']:.1f} Days", f"New CCC: {rebal_sim['optimized_outcomes']['optimized_ccc_days']:.1f} Days")
    with w3:
        st.metric("Annual Credit Interest Saved", f"${rebal_sim['optimized_outcomes']['annual_borrowing_interest_savings']:,.2f}")


# ----------------- TAB 22: STRATEGIC ACTION PLAN -----------------
with tab_recs:
    st.subheader("Strategic Business Action Plan & Projected Financial Return")
    recommendations = BusinessRecommendationEngine.get_strategic_recommendations()

    for rec in recommendations:
        with st.container():
            st.markdown(f"### {rec['id']} · {rec['title']}")
            col_r1, col_r2 = st.columns([3, 2])
            with col_r1:
                st.markdown(f"**Target Cohort:** `{rec['target_segment']}`")
                st.markdown(f"**Empirical Driver:** {rec['empirical_driver']}")
                st.markdown("**Actionable Initiatives:**")
                for item in rec["actionable_initiatives"]:
                    st.markdown(f"- {item}")
            with col_r2:
                st.success(f"**Projected Lift:** {rec['projected_financial_impact']}")
                st.info(f"**Primary KPI:** {rec['kpi_target']}")
            st.markdown("---")
