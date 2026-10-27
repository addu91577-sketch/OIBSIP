"""
Executive Retail Sales Analytics & Machine Learning Intelligence Dashboard.

Built with Streamlit and Plotly to provide interactive business intelligence,
time-series decomposition, demographic segmentation, Pareto inequality analysis,
and live real-time machine learning inference simulation.
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
    return analytics, ml


analytics, ml = load_engines()
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
        font-weight: bold;
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

# Header
st.markdown(
    '<div class="main-header">🛍️ Retail Sales Analytics & Predictive Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Executive Decision Support System · Exploratory Data Analysis & Production Machine Learning</div>',
    unsafe_allow_html=True,
)

# Sidebar Filters
st.sidebar.header("🔍 Global Dynamic Filters")
selected_category = st.sidebar.multiselect(
    "Product Category",
    options=sorted(df["Product Category"].unique()),
    default=sorted(df["Product Category"].unique()),
)
selected_gender = st.sidebar.multiselect(
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

# Filter Data
filtered_df = df[
    (df["Product Category"].isin(selected_category))
    & (df["Gender"].isin(selected_gender))
    & (df["Age"].between(age_range[0], age_range[1]))
]

st.sidebar.markdown("---")
st.sidebar.info(
    f"📊 **Active Cohort Records**: {len(filtered_df):,} / {len(df):,} ({(len(filtered_df)/len(df))*100:.1f}%)"
)

# Tabs
tab_overview, tab_timeseries, tab_demographics, tab_products, tab_pareto, tab_ml, tab_recs = (
    st.tabs(
        [
            "📊 Executive KPIs",
            "📈 Time-Series Trends",
            "👥 Customer Demographics",
            "🏷️ Merchandise & Pricing",
            "👑 Pareto Whale Economy",
            "🤖 AI / ML Simulator",
            "🎯 Strategic Action Plan",
        ]
    )
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
    st.subheader("Seasonal Cycles, Velocity & Weekday vs Weekend Dynamics")

    ts_data = analytics.get_time_series_analytics()
    monthly_df = pd.DataFrame(ts_data["monthly_trends"])
    dow_df = pd.DataFrame(ts_data["day_of_week_dynamics"]["daily_trends"])
    shift_df = pd.DataFrame(ts_data["day_of_week_dynamics"]["category_weekend_lifts"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 📅 Monthly Sales Velocity & Transaction Count")
        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Bar(
                x=monthly_df["YearMonth"],
                y=monthly_df["orders"],
                name="Transaction Orders",
                marker_color="#334155",
                yaxis="y2",
                opacity=0.6,
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=monthly_df["YearMonth"],
                y=monthly_df["total_revenue"],
                name="Total Revenue ($)",
                mode="lines+markers",
                marker_color="#38bdf8",
                line=dict(width=3),
            )
        )
        fig_ts.update_layout(
            template="plotly_dark",
            height=400,
            yaxis=dict(title="Revenue ($)", showgrid=False),
            yaxis2=dict(
                title="Orders",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_b:
        st.markdown("##### 🗓️ Day-of-Week Sales Volume vs Average Order Value")
        fig_dow = px.bar(
            dow_df,
            x="DayOfWeek",
            y="total_revenue",
            color="aov",
            text="aov",
            title="Daily Revenue Velocity & Average Order Spend",
            labels={"DayOfWeek": "Day", "total_revenue": "Revenue ($)", "aov": "AOV ($)"},
            color_continuous_scale="Viridis",
        )
        fig_dow.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
        fig_dow.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown("---")
    st.markdown("##### ⚡ The 'Weekend Lift' Analysis (Category Decoupling)")
    lift_cols = st.columns(3)
    for idx, row in shift_df.iterrows():
        with lift_cols[idx]:
            sign = "+" if row["weekend_lift_pct"] > 0 else ""
            st.metric(
                label=f"{row['category']} Weekend Lift",
                value=f"${row['weekend_aov']:,.2f}",
                delta=f"{sign}{row['weekend_lift_pct']}% vs Weekday (${row['weekday_aov']:,.2f})",
            )

# ----------------- TAB 3: CUSTOMER DEMOGRAPHICS -----------------
with tab_demographics:
    st.subheader("Demographic Segmentation & Customer Purchasing Personas")

    demographics = analytics.get_demographics_breakdown()
    cohort_df = pd.DataFrame(demographics["age_cohorts"])
    gender_df = pd.DataFrame(demographics["gender_breakdown"])

    c_demo1, c_demo2 = st.columns(2)
    with c_demo1:
        st.markdown("##### 🧑‍🤝‍🧑 Age Cohort Revenue Contribution vs Average Basket Spend")
        fig_cohort = px.bar(
            cohort_df,
            x="AgeCohort",
            y="total_revenue",
            color="aov",
            text="aov",
            title="Revenue Turnover & AOV Across Life Stages",
            color_continuous_scale="Blues",
            labels={"AgeCohort": "Cohort", "total_revenue": "Revenue ($)", "aov": "AOV ($)"},
        )
        fig_cohort.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
        fig_cohort.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_cohort, use_container_width=True)

    with c_demo2:
        st.markdown("##### 🚻 Gender Commercial Parity")
        fig_gen = px.pie(
            gender_df,
            values="total_revenue",
            names="Gender",
            hole=0.4,
            title="Revenue Distribution by Gender",
            color="Gender",
            color_discrete_map={"Female": "#ec4899", "Male": "#3b82f6"},
        )
        fig_gen.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_gen, use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
        > **💡 Demographic Persona Insights:**  
        > * **Young Adult Whale Paradox:** Patrons aged 18–25 convert to $1,000+ VIP orders at a **25.44% rate**, outperforming mature cohorts by +26%.  
        > * **Male Luxury Beauty Gifting:** Male shoppers spend **$487.13 AOV** in Beauty compared to $450.78 for female shoppers, surging to **$637.32 AOV** in young adult males (+51.8% premium).
        """
    )

# ----------------- TAB 4: MERCHANDISE & PRICING -----------------
with tab_products:
    st.subheader("Department Performance, Discrete Price Tiers & Best Sellers")

    prod_data = analytics.get_product_category_performance()
    tier_df = pd.DataFrame(prod_data["catalog_price_tiers"])
    top_df = pd.DataFrame(prod_data["top_offerings_by_revenue"])

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("##### 💎 The 88.4% Luxury Revenue Concentration")
        fig_tier = px.bar(
            tier_df,
            x="Price per Unit",
            y="total_revenue",
            text="revenue_share_pct",
            title="Total Revenue Generated by Discrete Catalog Price Tier",
            labels={"Price per Unit": "Catalog Unit Price ($)", "total_revenue": "Revenue ($)"},
            color="Price per Unit",
            color_continuous_scale="Purples",
        )
        fig_tier.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_tier.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_tier, use_container_width=True)

    with col_p2:
        st.markdown("##### 🏆 Top 10 Best-Selling Product Offerings")
        fig_top = px.bar(
            top_df,
            x="total_revenue",
            y="Product Offering",
            orientation="h",
            text="total_revenue",
            title="Top Offerings Ranked by Revenue Generated",
            labels={"total_revenue": "Total Spend ($)", "Product Offering": "Offering"},
            color="Product Category",
            color_discrete_map={
                "Electronics": "#38bdf8",
                "Clothing": "#ec4899",
                "Beauty": "#a855f7",
            },
        )
        fig_top.update_traces(texttemplate="$%{text:,.0f}", textposition="inside")
        fig_top.update_layout(
            template="plotly_dark", height=400, yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_top, use_container_width=True)

# ----------------- TAB 5: PARETO WHALE ECONOMY -----------------
with tab_pareto:
    st.subheader("Lorenz Curve Spending Inequality & VIP Concentration Analysis")

    pareto_data = analytics.get_pareto_concentration()
    lorenz_df = pd.DataFrame(pareto_data["lorenz_curve"])
    polar_df = pd.DataFrame(pareto_data["spend_tier_polarization"])
    dec_df = pd.DataFrame(pareto_data["decile_revenue_breakdown"])

    col_lorenz, col_polar = st.columns([3, 2])
    with col_lorenz:
        st.markdown(
            f"##### 📉 Empirical Lorenz Curve (Gini Index = **{pareto_data['gini_coefficient']}**)"
        )
        fig_lorenz = go.Figure()
        fig_lorenz.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Line of Perfect Equality",
                line=dict(dash="dash", color="#94a3b8"),
            )
        )
        fig_lorenz.add_trace(
            go.Scatter(
                x=lorenz_df["population_share"],
                y=lorenz_df["revenue_share"],
                mode="lines+markers",
                name="Actual Spending Concentration",
                line=dict(color="#f43f5e", width=3),
                fill="tonexty",
                fillcolor="rgba(244, 63, 94, 0.15)",
            )
        )
        fig_lorenz.update_layout(
            template="plotly_dark",
            height=420,
            xaxis=dict(title="Cumulative Share of Customer Transactions", range=[0, 1]),
            yaxis=dict(title="Cumulative Share of Store Revenue", range=[0, 1]),
        )
        st.plotly_chart(fig_lorenz, use_container_width=True)

    with col_polar:
        st.markdown("##### 🐋 Spend Tier Revenue Polarization")
        fig_pol = px.bar(
            polar_df,
            x="SpendTier",
            y=["order_share_pct", "revenue_share_pct"],
            barmode="group",
            title="Order Volume vs Revenue Contribution (%)",
            labels={"value": "Percentage (%)", "SpendTier": "Spend Tier"},
            color_discrete_map={
                "order_share_pct": "#64748b",
                "revenue_share_pct": "#f43f5e",
            },
        )
        fig_pol.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_pol, use_container_width=True)

# ----------------- TAB 6: AI / ML SIMULATOR -----------------
with tab_ml:
    st.subheader("Real-Time Machine Learning Inference & Scenario Simulator")
    st.markdown(
        "Interactively simulate customer transaction profiles to test VIP propensity, "
        "predict basket spend, and trigger dynamic cross-department bundle recommendations."
    )

    c_in1, c_in2, c_in3, c_in4, c_in5 = st.columns(5)
    with c_in1:
        sim_age = st.slider("Customer Age", 18, 65, 30)
    with c_in2:
        sim_gender = st.selectbox("Customer Gender", ["Female", "Male"])
    with c_in3:
        sim_cat = st.selectbox("Product Category", ["Electronics", "Clothing", "Beauty"])
    with c_in4:
        sim_qty = st.slider("Basket Quantity", 1, 4, 3)
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

# ----------------- TAB 7: STRATEGIC ACTION PLAN -----------------
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
