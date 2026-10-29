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
    return analytics, ml, drift, segmentation, sim, elasticity, ab_test, clv


analytics, ml, drift_engine, segmentation_engine, sim_engine, elasticity_engine, ab_engine, clv_engine = load_engines()
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
        "🎯 Strategic Action Plan",
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


# ----------------- TAB 13: STRATEGIC ACTION PLAN -----------------
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
