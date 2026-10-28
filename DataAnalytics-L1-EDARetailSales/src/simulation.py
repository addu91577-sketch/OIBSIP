"""
Financial What-If Scenario & Commercial Uplift Simulation Engine.

Evaluates strategic retail interventions:
1. Cross-Category Bundle Incentive (breaking the 2.51 unit basket inelasticity cap)
2. VIP Whale Retention & Repeat Transaction Escalation
3. Mid-Week Trough Flattening (Thursday Velocity Lift)
4. Comprehensive revenue, gross margin, discount allowance, and ROI projections.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.analytics import RetailAnalyticsEngine


class FinancialWhatIfSimulator:
    """
    Simulates commercial scenarios and projects revenue, margin, and ROI impact
    based on empirical baseline statistics.
    """

    def __init__(self, data_path: Optional[Path] = None, gross_margin_pct: float = 0.40):
        self.analytics = RetailAnalyticsEngine(data_path=data_path)
        self.df = self.analytics.df
        self.gross_margin_pct = gross_margin_pct  # Standard retail gross margin (40%)

        # Baseline metrics
        self.baseline_orders = len(self.df)
        self.baseline_revenue = float(self.df["Total Amount"].sum())
        self.baseline_units = float(self.df["Quantity"].sum())
        self.baseline_aov = self.baseline_revenue / self.baseline_orders
        self.baseline_avg_units = self.baseline_units / self.baseline_orders
        self.baseline_gross_profit = self.baseline_revenue * self.gross_margin_pct

        # VIP subset baselines
        self.vip_df = self.df[self.df["Total Amount"] >= 1000]
        self.vip_orders = len(self.vip_df)
        self.vip_revenue = float(self.vip_df["Total Amount"].sum())
        self.vip_aov = self.vip_revenue / self.vip_orders if self.vip_orders else 0.0

        # Day of week baselines
        dow_counts = self.df["DayOfWeek"].value_counts()
        self.thursday_orders = int(dow_counts.get("Thursday", 0))
        self.saturday_orders = int(dow_counts.get("Saturday", 0))

    def simulate_bundle_expansion(
        self,
        adoption_rate: float = 0.25,
        target_units: float = 3.20,
        bundle_discount_pct: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Simulates breaking the 2.51 unit cap with a tiered bundle discount.
        - adoption_rate: Fraction of shoppers adopting the bundle (e.g., 0.25 = 25%)
        - target_units: New basket unit count for adopters (baseline: 2.514)
        - bundle_discount_pct: Tiered discount given on bundle (e.g., 0.10 = 10%)
        """
        if not (0.0 <= adoption_rate <= 1.0):
            raise ValueError("adoption_rate must be between 0.0 and 1.0")
        if target_units <= 0:
            raise ValueError("target_units must be positive")
        if not (0.0 <= bundle_discount_pct < 1.0):
            raise ValueError("bundle_discount_pct must be between 0.0 and 0.99")

        adopting_orders = int(np.round(self.baseline_orders * adoption_rate))
        non_adopting_orders = self.baseline_orders - adopting_orders

        # Non-adopting orders retain baseline performance
        non_adopting_rev = non_adopting_orders * self.baseline_aov

        # Adopting orders scale units and receive discount
        expansion_factor = target_units / self.baseline_avg_units
        gross_basket_before_disc = self.baseline_aov * expansion_factor
        disc_amount_per_basket = gross_basket_before_disc * bundle_discount_pct
        net_basket_spend = gross_basket_before_disc - disc_amount_per_basket

        adopting_gross_rev = adopting_orders * gross_basket_before_disc
        adopting_disc_cost = adopting_orders * disc_amount_per_basket
        adopting_net_rev = adopting_orders * net_basket_spend

        projected_total_rev = non_adopting_rev + adopting_net_rev
        net_revenue_uplift = projected_total_rev - self.baseline_revenue
        pct_revenue_lift = (net_revenue_uplift / self.baseline_revenue) * 100

        # Profit & Margin Analysis based on target retail gross margin percentage
        projected_gross_profit = projected_total_rev * self.gross_margin_pct
        net_profit_uplift = projected_gross_profit - self.baseline_gross_profit
        program_roi = (net_profit_uplift / adopting_disc_cost * 100) if adopting_disc_cost > 0 else 0.0
        projected_total_units = (non_adopting_orders * self.baseline_avg_units) + (adopting_orders * target_units)

        return {
            "intervention": "Cross-Department Bundle Promotion",
            "parameters": {
                "adoption_rate": adoption_rate,
                "target_units": target_units,
                "bundle_discount_pct": bundle_discount_pct,
            },
            "financial_impact": {
                "baseline_revenue": self.baseline_revenue,
                "projected_revenue": float(np.round(projected_total_rev, 2)),
                "net_revenue_uplift": float(np.round(net_revenue_uplift, 2)),
                "pct_revenue_lift": float(np.round(pct_revenue_lift, 2)),
                "adopting_orders_count": adopting_orders,
                "discount_allowance_cost": float(np.round(adopting_disc_cost, 2)),
                "baseline_gross_profit": float(np.round(self.baseline_gross_profit, 2)),
                "projected_gross_profit": float(np.round(projected_gross_profit, 2)),
                "net_profit_uplift": float(np.round(net_profit_uplift, 2)),
                "program_roi_pct": float(np.round(program_roi, 1)),
                "baseline_aov": float(np.round(self.baseline_aov, 2)),
                "adopter_aov": float(np.round(net_basket_spend, 2)),
                "projected_units_sold": int(np.round(projected_total_units)),
            },
        }

    def simulate_vip_retention(
        self,
        retention_lift_pct: float = 0.15,
        loyalty_cost_pct: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Simulates VIP Whale retention and repeat visit escalation.
        - retention_lift_pct: Fraction of VIPs generating an extra transaction (e.g. 0.15 = 15%)
        - loyalty_cost_pct: Concierge / perk cost as fraction of incremental spend (e.g. 5%)
        """
        if not (0.0 <= retention_lift_pct <= 1.0):
            raise ValueError("retention_lift_pct must be between 0.0 and 1.0")

        incremental_vip_orders = int(np.round(self.vip_orders * retention_lift_pct))
        incremental_gross_rev = incremental_vip_orders * self.vip_aov
        loyalty_expense = incremental_gross_rev * loyalty_cost_pct
        incremental_cogs = incremental_gross_rev * (1 - self.gross_margin_pct)
        incremental_profit = incremental_gross_rev - incremental_cogs - loyalty_expense

        projected_rev = self.baseline_revenue + incremental_gross_rev
        pct_rev_lift = (incremental_gross_rev / self.baseline_revenue) * 100
        roi = (incremental_profit / loyalty_expense * 100) if loyalty_expense > 0 else 0.0

        return {
            "intervention": "VIP Concierge & Whale Retention",
            "parameters": {
                "retention_lift_pct": retention_lift_pct,
                "loyalty_cost_pct": loyalty_cost_pct,
            },
            "financial_impact": {
                "baseline_vip_orders": self.vip_orders,
                "incremental_vip_orders": incremental_vip_orders,
                "vip_aov": float(np.round(self.vip_aov, 2)),
                "incremental_revenue": float(np.round(incremental_gross_rev, 2)),
                "projected_revenue": float(np.round(projected_rev, 2)),
                "pct_revenue_lift": float(np.round(pct_rev_lift, 2)),
                "loyalty_expense": float(np.round(loyalty_expense, 2)),
                "incremental_profit": float(np.round(incremental_profit, 2)),
                "roi_pct": float(np.round(roi, 1)),
            },
        }

    def simulate_combined_strategy(
        self,
        bundle_adoption: float = 0.25,
        target_units: float = 3.20,
        bundle_discount: float = 0.10,
        vip_lift: float = 0.15,
    ) -> Dict[str, Any]:
        """Runs unified multi-lever scenario combining bundle promotion and VIP retention."""
        bundle_res = self.simulate_bundle_expansion(
            adoption_rate=bundle_adoption,
            target_units=target_units,
            bundle_discount_pct=bundle_discount,
        )
        vip_res = self.simulate_vip_retention(retention_lift_pct=vip_lift)

        bundle_fin = bundle_res["financial_impact"]
        vip_fin = vip_res["financial_impact"]

        total_incremental_rev = bundle_fin["net_revenue_uplift"] + vip_fin["incremental_revenue"]
        total_projected_rev = self.baseline_revenue + total_incremental_rev
        total_incremental_profit = bundle_fin["net_profit_uplift"] + vip_fin["incremental_profit"]
        total_costs = bundle_fin["discount_allowance_cost"] + vip_fin["loyalty_expense"]
        combined_roi = (total_incremental_profit / total_costs * 100) if total_costs > 0 else 0.0

        return {
            "strategy": "Unified Strategic Acceleration (Bundles + VIP Retention)",
            "baseline_revenue": self.baseline_revenue,
            "projected_total_revenue": float(np.round(total_projected_rev, 2)),
            "total_incremental_revenue": float(np.round(total_incremental_rev, 2)),
            "overall_pct_revenue_lift": float(np.round((total_incremental_rev / self.baseline_revenue) * 100, 2)),
            "total_program_cost": float(np.round(total_costs, 2)),
            "total_net_profit_gain": float(np.round(total_incremental_profit, 2)),
            "combined_roi_pct": float(np.round(combined_roi, 1)),
            "bundle_component": bundle_res,
            "vip_component": vip_res,
        }
