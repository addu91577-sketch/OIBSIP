"""
Customer Lifetime Value (CLV) & Churn Risk Probabilistic Modeling Engine.

Provides:
1. Probabilistic recency-based customer survival modeling: P(Active) = exp(-lambda * Recency)
2. Four-tier churn risk classification (Active, Cooling, At-Risk, Lapsed)
3. Discounted future Customer Lifetime Value (CLV) forecasting
4. Enterprise customer equity aggregation and cohort distribution
5. High-value VIP Whale churn early warning system with quantified revenue at risk
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.analytics import RetailAnalyticsEngine


class CustomerLifetimeValueEngine:
    """
    Probabilistic Customer Lifetime Value (CLV) and churn risk analytics
    grounded in retail transaction recency and spend dynamics.
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        decay_rate: float = 0.0075,
        gross_margin_pct: float = 0.40,
        annual_discount_rate: float = 0.10,
    ):
        self.analytics = RetailAnalyticsEngine(data_path=data_path)
        self.df = self.analytics.df.copy()
        self.decay_rate = decay_rate
        self.gross_margin_pct = gross_margin_pct
        self.annual_discount_rate = annual_discount_rate

        self._customer_df: Optional[pd.DataFrame] = None
        self._calculate_customer_metrics()

    def _calculate_customer_metrics(self) -> None:
        """
        Calculates individual customer recency, monetary metrics, churn probability,
        and projected 12-month CLV for all shoppers in the dataset.
        """
        ref_date = pd.to_datetime(self.df["Date"]).max()

        records = []
        for _, row in self.df.iterrows():
            tx_date = pd.to_datetime(row["Date"])
            recency_days = (ref_date - tx_date).days
            spend = float(row["Total Amount"])
            units = int(row["Quantity"])
            unit_price = float(row["Price per Unit"])
            category = row["Product Category"]
            gender = row["Gender"]
            age = int(row["Age"])
            cohort = row.get("AgeCohort", "Unknown")

            # Survival & Churn probability
            p_alive = float(np.exp(-self.decay_rate * recency_days))
            p_churn = float(1.0 - p_alive)

            # Churn Tier
            if p_churn < 0.35:
                churn_tier = "Active (Low Risk)"
            elif p_churn < 0.65:
                churn_tier = "Cooling (Medium Risk)"
            elif p_churn < 0.85:
                churn_tier = "At Risk (High Risk)"
            else:
                churn_tier = "Lapsed (Defected)"

            # Expected future transactions over 365 days
            # Benchmark baseline annual frequency = 1.0 purchase / year
            expected_annual_orders = p_alive * 1.5

            # Discounted future gross profit contribution (CLV)
            discount_factor = 1.0 / (1.0 + self.annual_discount_rate)
            expected_12m_clv = (
                expected_annual_orders * spend * self.gross_margin_pct * discount_factor
            )

            is_vip = bool(spend >= 1000.0)

            records.append(
                {
                    "customer_id": row["Customer ID"],
                    "transaction_id": int(row["Transaction ID"]),
                    "date": str(row["Date"]),
                    "recency_days": int(recency_days),
                    "historical_spend": spend,
                    "units": units,
                    "unit_price": unit_price,
                    "product_category": category,
                    "gender": gender,
                    "age": age,
                    "age_cohort": str(cohort),
                    "prob_alive": round(p_alive, 4),
                    "prob_churn": round(p_churn, 4),
                    "churn_tier": churn_tier,
                    "expected_annual_orders": round(expected_annual_orders, 3),
                    "projected_12m_clv": round(expected_12m_clv, 2),
                    "is_vip": is_vip,
                }
            )

        self._customer_df = pd.DataFrame(records)

    @property
    def customer_df(self) -> pd.DataFrame:
        if self._customer_df is None:
            self._calculate_customer_metrics()
        return self._customer_df

    def calculate_churn_probability(self, recency_days: float) -> float:
        """
        Computes conditional churn probability given recency in days.
        """
        if recency_days < 0:
            raise ValueError("recency_days cannot be negative.")
        p_alive = np.exp(-self.decay_rate * recency_days)
        return float(1.0 - p_alive)

    def predict_customer_clv(
        self,
        recency_days: float,
        aov: float,
        gross_margin_pct: Optional[float] = None,
        discount_rate_annual: Optional[float] = None,
        horizon_days: int = 365,
    ) -> Dict[str, Any]:
        """
        Predicts churn risk and projected discounted CLV for an arbitrary customer profile.
        """
        if recency_days < 0:
            raise ValueError("recency_days cannot be negative.")
        if aov <= 0:
            raise ValueError("aov must be strictly positive.")
        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive.")

        margin = gross_margin_pct if gross_margin_pct is not None else self.gross_margin_pct
        disc_rate = discount_rate_annual if discount_rate_annual is not None else self.annual_discount_rate

        p_alive = float(np.exp(-self.decay_rate * recency_days))
        p_churn = float(1.0 - p_alive)

        if p_churn < 0.35:
            churn_tier = "Active (Low Risk)"
            playbook = "Standard nurturing; upsell to multi-item cross-category bundles."
        elif p_churn < 0.65:
            churn_tier = "Cooling (Medium Risk)"
            playbook = "Re-engagement email campaign; personalized catalog highlight with 5% incentive."
        elif p_churn < 0.85:
            churn_tier = "At Risk (High Risk)"
            playbook = "Win-back trigger; exclusive 15% VIP reactivation offer before dormancy sets in."
        else:
            churn_tier = "Lapsed (Defected)"
            playbook = "Sunset protocol; targeted high-incentive quarterly re-acquisition or survey."

        annual_rate = horizon_days / 365.0
        expected_orders = p_alive * 1.5 * annual_rate
        discount_factor = 1.0 / (1.0 + disc_rate * annual_rate)
        expected_clv = expected_orders * aov * margin * discount_factor

        return {
            "recency_days": round(recency_days, 1),
            "aov": round(aov, 2),
            "horizon_days": horizon_days,
            "prob_alive": round(p_alive, 4),
            "prob_churn": round(p_churn, 4),
            "churn_tier": churn_tier,
            "expected_orders_in_horizon": round(expected_orders, 3),
            "projected_clv": round(expected_clv, 2),
            "recommended_playbook": playbook,
        }

    def audit_customer_equity(self) -> Dict[str, Any]:
        """
        Audits overall store customer equity, aggregate projected CLV,
        and churn distribution across tiers and demographic cohorts.
        """
        df = self.customer_df

        total_customers = len(df)
        total_historical_spend = float(df["historical_spend"].sum())
        total_projected_clv = float(df["projected_12m_clv"].sum())
        avg_projected_clv = total_projected_clv / total_customers if total_customers else 0.0

        # Risk Tier Breakdown
        tier_summary = []
        for tier_name, grp in df.groupby("churn_tier"):
            tier_clv = float(grp["projected_12m_clv"].sum())
            tier_spend = float(grp["historical_spend"].sum())
            tier_summary.append(
                {
                    "churn_tier": str(tier_name),
                    "customer_count": len(grp),
                    "pct_customers": round((len(grp) / total_customers) * 100, 2),
                    "avg_recency_days": round(float(grp["recency_days"].mean()), 1),
                    "avg_churn_prob_pct": round(float(grp["prob_churn"].mean()) * 100, 2),
                    "total_historical_spend": round(tier_spend, 2),
                    "total_projected_12m_clv": round(tier_clv, 2),
                    "pct_projected_clv": round((tier_clv / total_projected_clv) * 100, 2)
                    if total_projected_clv > 0
                    else 0.0,
                }
            )

        # VIP Churn Early Warning Audit
        vip_df = df[df["is_vip"]]
        vip_count = len(vip_df)
        vip_historical_spend = float(vip_df["historical_spend"].sum())
        vip_at_risk_df = vip_df[vip_df["churn_tier"].isin(["Cooling (Medium Risk)", "At Risk (High Risk)", "Lapsed (Defected)"])]
        vip_at_risk_spend = float(vip_at_risk_df["historical_spend"].sum())
        vip_at_risk_clv = float(vip_at_risk_df["projected_12m_clv"].sum())

        return {
            "customer_base": {
                "total_customers": total_customers,
                "total_historical_revenue": round(total_historical_spend, 2),
                "total_projected_12m_customer_equity": round(total_projected_clv, 2),
                "average_customer_clv": round(avg_projected_clv, 2),
            },
            "churn_tier_distribution": tier_summary,
            "vip_early_warning": {
                "total_vips": vip_count,
                "vip_historical_spend": round(vip_historical_spend, 2),
                "vips_at_risk_count": len(vip_at_risk_df),
                "vips_at_risk_pct": round((len(vip_at_risk_df) / vip_count) * 100, 2) if vip_count else 0.0,
                "vip_revenue_at_risk": round(vip_at_risk_spend, 2),
                "vip_projected_clv_at_risk": round(vip_at_risk_clv, 2),
            },
        }

    def get_at_risk_vips(self, min_spend: float = 1000.0, limit: int = 15) -> List[Dict[str, Any]]:
        """
        Returns high-priority VIP customer records who are in danger of churning.
        """
        df = self.customer_df
        vips = df[(df["historical_spend"] >= min_spend) & (df["prob_churn"] >= 0.50)]
        sorted_vips = vips.sort_values(by=["historical_spend", "prob_churn"], ascending=[False, False])

        results = []
        for _, row in sorted_vips.head(limit).iterrows():
            results.append(
                {
                    "customer_id": row["customer_id"],
                    "historical_spend": float(row["historical_spend"]),
                    "recency_days": int(row["recency_days"]),
                    "prob_churn_pct": round(float(row["prob_churn"]) * 100, 1),
                    "product_category": row["product_category"],
                    "age": int(row["age"]),
                    "gender": row["gender"],
                    "projected_12m_clv": float(row["projected_12m_clv"]),
                    "churn_tier": row["churn_tier"],
                }
            )
        return results
