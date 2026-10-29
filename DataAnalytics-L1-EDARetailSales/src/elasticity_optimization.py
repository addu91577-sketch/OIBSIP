"""
Price Elasticity of Demand (PED) & Commercial Revenue Optimization Engine.

Provides:
1. Econometric log-log price elasticity estimation: ln(Q) = alpha + beta * ln(P)
2. Empirical arc elasticity across catalog pricing tiers ($25, $30, $50, $300, $500)
3. Prescriptive markdown & promotional discount profitability evaluator
4. Theoretical and discrete optimal price point modeling under margin constraints
5. Demographic cohort elasticity cross-tabulation
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.analytics import RetailAnalyticsEngine


class PriceElasticityOptimizer:
    """
    Econometric elasticity estimation and pricing optimization engine
    for retail merchandise categories and customer demographic cohorts.
    """

    def __init__(self, data_path: Optional[Path] = None, gross_margin_pct: float = 0.40):
        self.analytics = RetailAnalyticsEngine(data_path=data_path)
        self.df = self.analytics.df.copy()
        self.gross_margin_pct = gross_margin_pct

    def compute_category_elasticity(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Computes econometric log-log and empirical arc price elasticity for a given category
        (or entire catalog if category is None).
        """
        data = self.df if category is None else self.df[self.df["Product Category"] == category]

        if len(data) == 0:
            raise ValueError(f"Category '{category}' has no transaction records.")

        # Aggregate quantity and average unit price per discrete tier
        tier_agg = (
            data.groupby("Price per Unit")
            .agg(
                order_count=("Transaction ID", "count"),
                total_units=("Quantity", "sum"),
                avg_units_per_order=("Quantity", "mean"),
                total_revenue=("Total Amount", "sum"),
            )
            .reset_index()
            .sort_values(by="Price per Unit")
        )

        prices = tier_agg["Price per Unit"].values.astype(float)
        quantities = tier_agg["avg_units_per_order"].values.astype(float)

        # 1. Log-Log Econometric Regression: ln(Q) = alpha + beta * ln(P)
        log_p = np.log(prices)
        log_q = np.log(quantities)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_p, log_q)
        ped_coefficient = float(slope)
        r_squared = float(r_value**2)

        # 2. Arc Elasticity between adjacent tiers:
        # E_arc = ((Q2 - Q1) / ((Q1 + Q2) / 2)) / ((P2 - P1) / ((P1 + P2) / 2))
        arc_elasticities = []
        for i in range(len(prices) - 1):
            p1, p2 = prices[i], prices[i + 1]
            q1, q2 = quantities[i], quantities[i + 1]

            pct_delta_p = (p2 - p1) / ((p1 + p2) / 2.0)
            pct_delta_q = (q2 - q1) / ((q1 + q2) / 2.0)

            arc_e = float(pct_delta_q / pct_delta_p) if pct_delta_p != 0 else 0.0
            arc_elasticities.append(
                {
                    "from_tier": float(p1),
                    "to_tier": float(p2),
                    "arc_elasticity": round(arc_e, 4),
                }
            )

        # 3. Elasticity Classification
        abs_ped = abs(ped_coefficient)
        if abs_ped < 0.2:
            classification = "Strictly Inelastic"
            interpretation = "Quantity per basket is virtually unaffected by price; revenue scales linearly with price point."
        elif abs_ped < 0.8:
            classification = "Inelastic"
            interpretation = "Demand responds modestly to price changes; price increases yield net revenue expansions."
        elif abs_ped <= 1.2:
            classification = "Unitary Elastic"
            interpretation = "Percentage price changes produce equal and opposite volume changes; revenue remains stable."
        else:
            classification = "Elastic"
            interpretation = "Volume is hyper-sensitive to price; price cuts can unlock significant volume expansion."

        # 4. Breakeven Volume Lift for Markdowns
        # Breakeven volume lift = d / (margin - d)
        markdown_discounts = [0.05, 0.10, 0.15, 0.20, 0.25]
        breakeven_lifts = []
        for d in markdown_discounts:
            if self.gross_margin_pct > d:
                be_lift = d / (self.gross_margin_pct - d)
                breakeven_lifts.append(
                    {
                        "discount_pct": round(d * 100, 1),
                        "breakeven_volume_lift_pct": round(be_lift * 100, 2),
                    }
                )

        return {
            "category": category or "All Categories",
            "transaction_count": int(len(data)),
            "ped_coefficient": round(ped_coefficient, 4),
            "r_squared": round(r_squared, 4),
            "p_value": round(float(p_value), 4),
            "classification": classification,
            "interpretation": interpretation,
            "price_tiers": tier_agg.to_dict(orient="records"),
            "arc_elasticities": arc_elasticities,
            "breakeven_markdown_requirements": breakeven_lifts,
        }

    def get_overall_elasticity_summary(self) -> Dict[str, Any]:
        """
        Returns a comparative summary of price elasticity across all 3 catalog categories.
        """
        categories = sorted(self.df["Product Category"].unique().tolist())
        category_summaries = {}

        for cat in categories:
            category_summaries[cat] = self.compute_category_elasticity(category=cat)

        overall = self.compute_category_elasticity(category=None)

        return {
            "catalog_overall": overall,
            "categories": category_summaries,
            "baseline_gross_margin_pct": self.gross_margin_pct,
        }

    def simulate_markdown_pricing(
        self,
        category: str,
        discount_pct: float,
        expected_volume_lift_pct: float,
        gross_margin_pct: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates the financial impact of a promotional markdown discount on a retail category.
        Determines whether volume lift covers margin sacrifice.
        """
        if not (0.0 <= discount_pct < 1.0):
            raise ValueError("discount_pct must be between 0.0 and 0.99")
        if expected_volume_lift_pct < -1.0:
            raise ValueError("expected_volume_lift_pct cannot be less than -1.0 (-100%)")

        margin = gross_margin_pct if gross_margin_pct is not None else self.gross_margin_pct

        cat_df = self.df[self.df["Product Category"] == category]
        if len(cat_df) == 0:
            raise ValueError(f"Category '{category}' does not exist in catalog.")

        base_orders = len(cat_df)
        base_units = float(cat_df["Quantity"].sum())
        base_revenue = float(cat_df["Total Amount"].sum())
        base_gross_profit = base_revenue * margin
        base_aov = base_revenue / base_orders if base_orders else 0.0

        # Post-promotion metrics
        # New unit volume
        new_units = base_units * (1.0 + expected_volume_lift_pct)
        # Average effective price after discount
        base_effective_price_per_unit = base_revenue / base_units if base_units else 0.0
        post_discount_price_per_unit = base_effective_price_per_unit * (1.0 - discount_pct)

        new_revenue = new_units * post_discount_price_per_unit

        # Cost per unit = baseline effective price * (1 - margin)
        unit_cost = base_effective_price_per_unit * (1.0 - margin)
        new_gross_profit = new_revenue - (new_units * unit_cost)

        revenue_delta = new_revenue - base_revenue
        revenue_delta_pct = (revenue_delta / base_revenue) * 100 if base_revenue else 0.0

        profit_delta = new_gross_profit - base_gross_profit
        profit_delta_pct = (profit_delta / base_gross_profit) * 100 if base_gross_profit else 0.0

        post_promo_margin_pct = (new_gross_profit / new_revenue) if new_revenue else 0.0

        # Required breakeven lift
        breakeven_lift_pct = (
            (discount_pct / (margin - discount_pct)) * 100
            if margin > discount_pct
            else float("inf")
        )

        # Strategic decision
        if profit_delta > 0:
            status = "Value Accretive"
            guidance = "Volume expansion outpaces margin concession; promotion generates net commercial profit."
        elif revenue_delta > 0:
            status = "Revenue Accretive / Margin Dilutive"
            guidance = "Total gross receipts increase, but gross profit declines due to inadequate volume compensation."
        else:
            status = "Value Destructive"
            guidance = "Both top-line revenue and bottom-line margin contract; avoid this discount structure."

        return {
            "category": category,
            "baseline": {
                "orders": base_orders,
                "units": round(base_units, 2),
                "revenue": round(base_revenue, 2),
                "gross_profit": round(base_gross_profit, 2),
                "effective_aov": round(base_aov, 2),
                "gross_margin_pct": round(margin * 100, 2),
            },
            "post_promotion": {
                "discount_pct": round(discount_pct * 100, 2),
                "volume_lift_pct": round(expected_volume_lift_pct * 100, 2),
                "simulated_units": round(new_units, 2),
                "simulated_revenue": round(new_revenue, 2),
                "simulated_gross_profit": round(new_gross_profit, 2),
                "post_promo_margin_pct": round(post_promo_margin_pct * 100, 2),
            },
            "financial_impact": {
                "revenue_delta": round(revenue_delta, 2),
                "revenue_delta_pct": round(revenue_delta_pct, 2),
                "profit_delta": round(profit_delta, 2),
                "profit_delta_pct": round(profit_delta_pct, 2),
                "breakeven_volume_lift_required_pct": round(breakeven_lift_pct, 2),
                "status": status,
                "executive_guidance": guidance,
            },
        }

    def optimize_price_points(
        self,
        category: str,
        target_margin_pct: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Simulates expected revenue and profit contributions across catalog discrete pricing tiers
        [$25, $30, $50, $300, $500] and identifies profit-maximizing tier.
        """
        margin = target_margin_pct if target_margin_pct is not None else self.gross_margin_pct
        cat_df = self.df[self.df["Product Category"] == category]

        if len(cat_df) == 0:
            raise ValueError(f"Category '{category}' does not exist.")

        tier_summary = []
        for p, grp in cat_df.groupby("Price per Unit"):
            rev = float(grp["Total Amount"].sum())
            orders = len(grp)
            units = float(grp["Quantity"].sum())
            unit_cost = float(p) * (1.0 - margin)
            gross_profit = rev - (units * unit_cost)
            profit_per_order = gross_profit / orders if orders else 0.0

            tier_summary.append(
                {
                    "price_tier": float(p),
                    "orders": orders,
                    "total_units": units,
                    "avg_basket_units": round(units / orders, 2),
                    "total_revenue": round(rev, 2),
                    "gross_profit": round(gross_profit, 2),
                    "profit_per_order": round(profit_per_order, 2),
                }
            )

        tier_df = pd.DataFrame(tier_summary)
        max_revenue_tier = float(tier_df.loc[tier_df["total_revenue"].idxmax(), "price_tier"])
        max_profit_tier = float(tier_df.loc[tier_df["gross_profit"].idxmax(), "price_tier"])

        return {
            "category": category,
            "target_margin_pct": margin,
            "tier_evaluations": tier_summary,
            "revenue_maximizing_tier": max_revenue_tier,
            "profit_maximizing_tier": max_profit_tier,
            "recommendation": (
                f"For {category}, the ${int(max_profit_tier)} price tier yields maximum cumulative gross "
                f"profit (${tier_df['gross_profit'].max():,.2f}) and represents the core margin pillar."
            ),
        }

    def compute_cohort_elasticity(self) -> List[Dict[str, Any]]:
        """
        Calculates log-log price elasticity broken down by customer age demographic cohorts.
        """
        cohort_col = "AgeCohort" if "AgeCohort" in self.df.columns else "Age Cohort"
        cohort_results = []
        for cohort, grp in self.df.groupby(cohort_col, observed=True):
            if len(grp) < 10:
                continue

            tier_agg = (
                grp.groupby("Price per Unit")
                .agg(avg_units=("Quantity", "mean"))
                .reset_index()
            )

            if len(tier_agg) >= 3:
                log_p = np.log(tier_agg["Price per Unit"].values)
                log_q = np.log(tier_agg["avg_units"].values)
                slope, _, _, p_val, _ = stats.linregress(log_p, log_q)

                cohort_results.append(
                    {
                        "age_cohort": str(cohort),
                        "sample_size": len(grp),
                        "ped_slope": round(float(slope), 4),
                        "p_value": round(float(p_val), 4),
                        "classification": "Inelastic" if abs(slope) < 0.5 else "Responsive",
                    }
                )

        return cohort_results
