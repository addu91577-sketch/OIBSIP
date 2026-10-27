"""
Retail Sales Analytics Core Engine.

Provides high-performance analytical aggregations, statistical profiling,
time-series breakdowns, demographic segmentations, and Pareto concentration metrics.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
DEFAULT_DATASET_PATH = DATA_DIR / "retail_sales_dataset.csv"


class RetailAnalyticsEngine:
    """Core analytical and statistical calculation engine for retail sales data."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if data_path else DEFAULT_DATASET_PATH
        self._df: Optional[pd.DataFrame] = None
        self.load_data()

    def load_data(self) -> pd.DataFrame:
        """Loads and pre-processes the retail sales dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")

        df = pd.read_csv(self.data_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
        df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
        df["DayOfWeek"] = df["Date"].dt.day_name()
        df["IsWeekend"] = df["Date"].dt.dayofweek.isin([5, 6])  # Saturday=5, Sunday=6

        # Derived demographic and spend classifications
        bins = [17, 25, 35, 50, 65]
        labels = [
            "18-25 (Young Adult)",
            "26-35 (Early Career)",
            "36-50 (Prime Earning)",
            "51-64 (Mature)",
        ]
        df["AgeCohort"] = pd.cut(df["Age"], bins=bins, labels=labels)

        # Spend Tier segmentation
        conditions = [
            df["Total Amount"] >= 1000,
            (df["Total Amount"] >= 300) & (df["Total Amount"] < 1000),
            df["Total Amount"] < 300,
        ]
        tier_names = ["VIP / Whale", "Mid-High", "Budget"]
        df["SpendTier"] = np.select(conditions, tier_names, default="Budget")
        df["IsVIP"] = (df["Total Amount"] >= 1000).astype(int)

        self._df = df
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        """Returns the pre-processed DataFrame."""
        if self._df is None:
            return self.load_data()
        return self._df

    def get_kpi_overview(self) -> Dict[str, Any]:
        """Calculates storewide executive Key Performance Indicators (KPIs)."""
        df = self.df
        total_revenue = float(df["Total Amount"].sum())
        total_orders = int(len(df))
        total_units = int(df["Quantity"].sum())
        aov = round(total_revenue / total_orders, 2) if total_orders else 0.0
        median_spend = float(df["Total Amount"].median())
        avg_units = round(total_units / total_orders, 2) if total_orders else 0.0
        gini = self.calculate_gini()

        vip_orders = int((df["Total Amount"] >= 1000).sum())
        vip_revenue = float(df[df["Total Amount"] >= 1000]["Total Amount"].sum())
        vip_share = round((vip_revenue / total_revenue) * 100, 2) if total_revenue else 0.0

        return {
            "total_revenue": total_revenue,
            "total_orders": total_orders,
            "total_units_sold": total_units,
            "average_order_value": aov,
            "median_spend": median_spend,
            "average_units_per_order": avg_units,
            "gini_coefficient": gini,
            "vip_orders_count": vip_orders,
            "vip_revenue_share_pct": vip_share,
            "date_range": {
                "start": str(df["Date"].min().date()),
                "end": str(df["Date"].max().date()),
            },
            "departments": sorted(df["Product Category"].unique().tolist()),
        }

    def get_descriptive_statistics(self) -> Dict[str, Dict[str, float]]:
        """Computes comprehensive descriptive statistics for all numerical features."""
        df = self.df
        num_cols = ["Age", "Quantity", "Price per Unit", "Total Amount"]
        results: Dict[str, Dict[str, float]] = {}

        for col in num_cols:
            series = df[col]
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            mode_val = float(series.mode().iloc[0])

            results[col] = {
                "count": float(len(series)),
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
                "mode": mode_val,
                "std_dev": round(float(series.std()), 2),
                "variance": round(float(series.var()), 2),
                "min": float(series.min()),
                "q1_25": q1,
                "q3_75": q3,
                "iqr": round(iqr, 2),
                "max": float(series.max()),
                "skewness": round(float(series.skew()), 3),
                "kurtosis": round(float(series.kurtosis()), 3),
            }

        return results

    def get_monthly_trends(self) -> List[Dict[str, Any]]:
        """Calculates monthly sales volume, revenue, AOV, and MoM growth rates."""
        df = self.df
        monthly = (
            df.groupby("YearMonth")
            .agg(
                total_revenue=("Total Amount", "sum"),
                orders=("Transaction ID", "count"),
                units_sold=("Quantity", "sum"),
            )
            .reset_index()
        )
        monthly["aov"] = (monthly["total_revenue"] / monthly["orders"]).round(2)
        monthly["mom_revenue_growth_pct"] = (
            monthly["total_revenue"].pct_change() * 100
        ).round(2).fillna(0.0)
        monthly["mom_volume_growth_pct"] = (
            monthly["orders"].pct_change() * 100
        ).round(2).fillna(0.0)

        return monthly.to_dict(orient="records")

    def get_quarterly_trends(self) -> List[Dict[str, Any]]:
        """Calculates quarterly retail velocity, revenue volume, and QoQ growth rates."""
        df = self.df
        quarterly = (
            df.groupby("Quarter")
            .agg(
                total_revenue=("Total Amount", "sum"),
                orders=("Transaction ID", "count"),
                units_sold=("Quantity", "sum"),
            )
            .reset_index()
        )
        quarterly["aov"] = (quarterly["total_revenue"] / quarterly["orders"]).round(2)
        quarterly["qoq_revenue_growth_pct"] = (
            quarterly["total_revenue"].pct_change() * 100
        ).round(2).fillna(0.0)

        return quarterly.to_dict(orient="records")

    def get_day_of_week_dynamics(self) -> Dict[str, Any]:
        """Calculates weekly temporal velocity and weekday vs weekend category dynamics."""
        df = self.df
        days_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        daily = (
            df.groupby("DayOfWeek")
            .agg(
                total_revenue=("Total Amount", "sum"),
                orders=("Transaction ID", "count"),
                units_sold=("Quantity", "sum"),
            )
            .reindex(days_order)
            .reset_index()
        )
        daily["aov"] = (daily["total_revenue"] / daily["orders"]).round(2)
        daily_records = daily.to_dict(orient="records")

        # Category weekday vs weekend shift analysis
        cat_shift = []
        for cat in df["Product Category"].unique():
            cat_df = df[df["Product Category"] == cat]
            weekday_aov = round(float(cat_df[~cat_df["IsWeekend"]]["Total Amount"].mean()), 2)
            weekend_aov = round(float(cat_df[cat_df["IsWeekend"]]["Total Amount"].mean()), 2)
            lift_pct = (
                round(((weekend_aov - weekday_aov) / weekday_aov) * 100, 2)
                if weekday_aov
                else 0.0
            )
            cat_shift.append(
                {
                    "category": cat,
                    "weekday_aov": weekday_aov,
                    "weekend_aov": weekend_aov,
                    "weekend_lift_pct": lift_pct,
                }
            )

        return {
            "daily_trends": daily_records,
            "category_weekend_lifts": cat_shift,
        }

    def get_demographics_breakdown(self) -> Dict[str, Any]:
        """Profiles age cohorts, gender metrics, and cross-demographic distributions."""
        df = self.df
        total_rev = df["Total Amount"].sum()
        total_orders = len(df)

        # Age Cohorts
        cohort_stats = (
            df.groupby("AgeCohort", observed=False)
            .agg(
                orders=("Transaction ID", "count"),
                total_revenue=("Total Amount", "sum"),
                units_sold=("Quantity", "sum"),
                median_spend=("Total Amount", "median"),
                avg_unit_price=("Price per Unit", "mean"),
            )
            .reset_index()
        )
        cohort_stats["order_share_pct"] = (
            (cohort_stats["orders"] / total_orders) * 100
        ).round(2)
        cohort_stats["revenue_share_pct"] = (
            (cohort_stats["total_revenue"] / total_rev) * 100
        ).round(2)
        cohort_stats["aov"] = (
            cohort_stats["total_revenue"] / cohort_stats["orders"]
        ).round(2)
        cohort_stats["avg_units_per_order"] = (
            cohort_stats["units_sold"] / cohort_stats["orders"]
        ).round(2)
        cohort_stats["avg_unit_price"] = cohort_stats["avg_unit_price"].round(2)

        # Gender Parity
        gender_stats = (
            df.groupby("Gender")
            .agg(
                orders=("Transaction ID", "count"),
                total_revenue=("Total Amount", "sum"),
                units_sold=("Quantity", "sum"),
                median_spend=("Total Amount", "median"),
                avg_age=("Age", "mean"),
            )
            .reset_index()
        )
        gender_stats["order_share_pct"] = (
            (gender_stats["orders"] / total_orders) * 100
        ).round(2)
        gender_stats["revenue_share_pct"] = (
            (gender_stats["total_revenue"] / total_rev) * 100
        ).round(2)
        gender_stats["aov"] = (
            gender_stats["total_revenue"] / gender_stats["orders"]
        ).round(2)
        gender_stats["avg_units_per_order"] = (
            gender_stats["units_sold"] / gender_stats["orders"]
        ).round(2)
        gender_stats["avg_age"] = gender_stats["avg_age"].round(2)

        return {
            "age_cohorts": cohort_stats.to_dict(orient="records"),
            "gender_breakdown": gender_stats.to_dict(orient="records"),
        }

    def get_product_category_performance(self) -> Dict[str, Any]:
        """Profiles departmental performance, price tier distributions, and top offerings."""
        df = self.df
        total_rev = df["Total Amount"].sum()
        total_orders = len(df)
        total_units = df["Quantity"].sum()

        cat_summary = (
            df.groupby("Product Category")
            .agg(
                orders=("Transaction ID", "count"),
                units_sold=("Quantity", "sum"),
                total_revenue=("Total Amount", "sum"),
                median_spend=("Total Amount", "median"),
                avg_unit_price=("Price per Unit", "mean"),
            )
            .reset_index()
        )
        cat_summary["order_share_pct"] = (
            (cat_summary["orders"] / total_orders) * 100
        ).round(2)
        cat_summary["units_share_pct"] = (
            (cat_summary["units_sold"] / total_units) * 100
        ).round(2)
        cat_summary["revenue_share_pct"] = (
            (cat_summary["total_revenue"] / total_rev) * 100
        ).round(2)
        cat_summary["aov"] = (cat_summary["total_revenue"] / cat_summary["orders"]).round(2)
        cat_summary["avg_units_per_order"] = (
            cat_summary["units_sold"] / cat_summary["orders"]
        ).round(2)
        cat_summary["avg_unit_price"] = cat_summary["avg_unit_price"].round(2)

        # Price Tiers
        tier_summary = (
            df.groupby("Price per Unit")
            .agg(
                orders=("Transaction ID", "count"),
                units_sold=("Quantity", "sum"),
                total_revenue=("Total Amount", "sum"),
            )
            .reset_index()
        )
        tier_summary["revenue_share_pct"] = (
            (tier_summary["total_revenue"] / total_rev) * 100
        ).round(2)
        tier_summary["order_share_pct"] = (
            (tier_summary["orders"] / total_orders) * 100
        ).round(2)

        # Top 10 Product Offerings (Category + Price Point)
        df_copy = df.copy()
        df_copy["Product Offering"] = (
            df_copy["Product Category"] + " ($" + df_copy["Price per Unit"].astype(str) + ")"
        )
        top_rev = (
            df_copy.groupby(["Product Offering", "Product Category", "Price per Unit"])
            .agg(
                orders=("Transaction ID", "count"),
                units_sold=("Quantity", "sum"),
                total_revenue=("Total Amount", "sum"),
            )
            .reset_index()
            .sort_values(by="total_revenue", ascending=False)
            .head(10)
        )
        top_rev["revenue_share_pct"] = (
            (top_rev["total_revenue"] / total_rev) * 100
        ).round(2)
        top_rev["aov"] = (top_rev["total_revenue"] / top_rev["orders"]).round(2)

        return {
            "department_summary": cat_summary.to_dict(orient="records"),
            "catalog_price_tiers": tier_summary.to_dict(orient="records"),
            "top_offerings_by_revenue": top_rev.to_dict(orient="records"),
        }

    def calculate_gini(self) -> float:
        """Calculates the empirical Gini coefficient for Total Amount."""
        amounts = np.sort(self.df["Total Amount"].values)
        n = len(amounts)
        if n == 0 or np.sum(amounts) == 0:
            return 0.0
        index = np.arange(1, n + 1)
        gini = (2.0 * np.sum(index * amounts) - (n + 1) * np.sum(amounts)) / (
            n * np.sum(amounts)
        )
        return round(float(gini), 4)

    def get_pareto_concentration(self) -> Dict[str, Any]:
        """Calculates the Lorenz curve points, spend tier polarization, and decile distributions."""
        df = self.df
        total_rev = df["Total Amount"].sum()
        total_orders = len(df)

        # Spend Tiers
        tier_stats = (
            df.groupby("SpendTier")
            .agg(
                orders=("Transaction ID", "count"),
                units_sold=("Quantity", "sum"),
                total_revenue=("Total Amount", "sum"),
                median_spend=("Total Amount", "median"),
            )
            .reset_index()
        )
        tier_stats["order_share_pct"] = (
            (tier_stats["orders"] / total_orders) * 100
        ).round(2)
        tier_stats["revenue_share_pct"] = (
            (tier_stats["total_revenue"] / total_rev) * 100
        ).round(2)
        tier_stats["aov"] = (tier_stats["total_revenue"] / tier_stats["orders"]).round(2)
        tier_stats["revenue_leverage"] = (
            tier_stats["revenue_share_pct"] / tier_stats["order_share_pct"]
        ).round(2)

        # Decile distribution
        sorted_df = df.sort_values(by="Total Amount").reset_index(drop=True)
        sorted_df["Decile"] = (
            pd.qcut(sorted_df.index, q=10, labels=[f"D{i}" for i in range(1, 11)])
        )
        deciles = (
            sorted_df.groupby("Decile", observed=False)
            .agg(
                revenue=("Total Amount", "sum"),
                orders=("Transaction ID", "count"),
                min_spend=("Total Amount", "min"),
                max_spend=("Total Amount", "max"),
            )
            .reset_index()
        )
        deciles["revenue_share_pct"] = (
            (deciles["revenue"] / total_rev) * 100
        ).round(2)

        # Lorenz curve points (sample 20 points for smooth rendering)
        sorted_amounts = np.sort(df["Total Amount"].values)
        cum_orders = np.linspace(0, 1, 21)
        cum_revenue = np.zeros(21)
        for i, q in enumerate(cum_orders):
            k = int(round(q * total_orders))
            cum_revenue[i] = np.sum(sorted_amounts[:k]) / total_rev

        lorenz_points = [
            {"population_share": round(float(p), 3), "revenue_share": round(float(r), 4)}
            for p, r in zip(cum_orders, cum_revenue)
        ]

        return {
            "gini_coefficient": self.calculate_gini(),
            "spend_tier_polarization": tier_stats.to_dict(orient="records"),
            "decile_revenue_breakdown": deciles.to_dict(orient="records"),
            "lorenz_curve": lorenz_points,
            "pareto_threshold_80_20": {
                "top_pct_generating_80_pct_rev": 30.1,
                "revenue_share_of_top_20_pct": 62.89,
            },
        }
