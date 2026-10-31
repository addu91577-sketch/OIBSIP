"""
Supply Chain & Inventory Optimization Engine for Retail Operations.

Provides:
1. Daily demand aggregation and statistical velocity profiling (mean, variance, daily std).
2. Economic Order Quantity (EOQ) and total inventory holding cost minimization.
3. Dynamic Safety Stock (SS) and Reorder Point (ROP) calibration under lead time uncertainty.
4. Stockout probability, expected lost units, and gross margin at risk simulation.
5. Multi-criteria ABC-FSN strategic inventory stratification (revenue impact vs sales velocity).
6. Enterprise working capital allocation and service-level sensitivity modeling.
"""

import math
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.analytics import RetailAnalyticsEngine


class InventorySupplyChainEngine:
    """
    Enterprise retail supply chain optimization engine.
    Calculates safety stock, ROP, EOQ, stockout risks, and ABC-FSN inventory stratification.
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        default_holding_cost_rate: float = 0.20,
        default_order_cost: float = 50.0,
        default_lead_time_days: float = 7.0,
        default_lead_time_std_days: float = 1.5,
    ):
        self.analytics = RetailAnalyticsEngine(data_path=data_path)
        self.df = self.analytics.df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.default_holding_cost_rate = default_holding_cost_rate
        self.default_order_cost = default_order_cost
        self.default_lead_time_days = default_lead_time_days
        self.default_lead_time_std_days = default_lead_time_std_days

    def get_daily_demand_metrics(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates daily demand statistics, velocity, and average unit price
        for a specified merchandise category or catalog-wide.
        """
        data = self.df if category is None else self.df[self.df["Product Category"] == category]

        if len(data) == 0:
            raise ValueError(f"Category '{category}' has no transaction records.")

        # Aggregate daily demand across calendar dates
        daily_units = data.groupby("Date")["Quantity"].sum()
        daily_rev = data.groupby("Date")["Total Amount"].sum()

        total_units = int(daily_units.sum())
        total_revenue = float(daily_rev.sum())
        active_days = int(len(daily_units))
        daily_mean = float(daily_units.mean())
        daily_std = float(daily_units.std(ddof=1)) if active_days > 1 else 0.0
        avg_unit_price = float(data["Price per Unit"].mean())

        return {
            "category": category if category else "All Categories",
            "total_units": total_units,
            "total_revenue": round(total_revenue, 2),
            "active_days": active_days,
            "daily_mean_units": round(daily_mean, 3),
            "daily_std_units": round(daily_std, 3),
            "daily_variance": round(daily_std**2, 3),
            "min_daily_units": int(daily_units.min()),
            "max_daily_units": int(daily_units.max()),
            "avg_unit_price": round(avg_unit_price, 2),
        }

    def calculate_safety_stock_and_rop(
        self,
        service_level: float = 0.95,
        lead_time_days: Optional[float] = None,
        lead_time_std: Optional[float] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculates Safety Stock (SS) and Reorder Point (ROP) under stochastic demand
        and variable lead times: SS = Z * sqrt(L * sigma_d^2 + d^2 * sigma_L^2).
        """
        if not (0.50 <= service_level < 1.0):
            raise ValueError(f"Service level must be in [0.50, 1.0), got {service_level}")

        l_days = lead_time_days if lead_time_days is not None else self.default_lead_time_days
        l_std = lead_time_std if lead_time_std is not None else self.default_lead_time_std_days

        if l_days <= 0 or l_std < 0:
            raise ValueError("Lead time must be positive and lead time std non-negative.")

        metrics = self.get_daily_demand_metrics(category=category)
        d_mean = metrics["daily_mean_units"]
        d_std = metrics["daily_std_units"]
        unit_price = metrics["avg_unit_price"]

        # Standard normal safety factor (Z-score)
        z_score = float(stats.norm.ppf(service_level))

        # Standard deviation of demand during lead time (LTD)
        # Var(LTD) = L * Var(d) + d^2 * Var(L)
        ltd_var = (l_days * (d_std**2)) + ((d_mean**2) * (l_std**2))
        ltd_std = math.sqrt(max(0.0, ltd_var))
        ltd_mean = d_mean * l_days

        safety_stock = z_score * ltd_std
        reorder_point = ltd_mean + safety_stock
        safety_stock_capital = safety_stock * unit_price

        return {
            "category": metrics["category"],
            "service_level": round(service_level, 4),
            "z_score": round(z_score, 3),
            "lead_time_days": round(l_days, 2),
            "lead_time_std_days": round(l_std, 2),
            "lead_time_demand_mean": round(ltd_mean, 2),
            "lead_time_demand_std": round(ltd_std, 3),
            "safety_stock_units": int(math.ceil(safety_stock)),
            "safety_stock_units_exact": round(safety_stock, 2),
            "reorder_point_units": int(math.ceil(reorder_point)),
            "reorder_point_units_exact": round(reorder_point, 2),
            "safety_stock_capital": round(safety_stock_capital, 2),
            "daily_mean_demand": round(d_mean, 2),
            "daily_std_demand": round(d_std, 2),
        }

    def calculate_eoq(
        self,
        order_cost: Optional[float] = None,
        holding_cost_rate: Optional[float] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculates Economic Order Quantity (EOQ), order cycles, and inventory holding costs:
        EOQ = sqrt((2 * D * S) / H).
        """
        s_cost = order_cost if order_cost is not None else self.default_order_cost
        h_rate = holding_cost_rate if holding_cost_rate is not None else self.default_holding_cost_rate

        if s_cost <= 0 or h_rate <= 0:
            raise ValueError("Order cost and holding cost rate must be strictly positive.")

        metrics = self.get_daily_demand_metrics(category=category)
        annual_demand = metrics["total_units"]
        unit_price = metrics["avg_unit_price"]

        # Unit holding cost per year
        unit_holding_cost = h_rate * unit_price

        if unit_holding_cost <= 0:
            unit_holding_cost = 1.0

        eoq = math.sqrt((2.0 * annual_demand * s_cost) / unit_holding_cost)
        annual_orders = annual_demand / eoq if eoq > 0 else 0.0
        cycle_days = 365.0 / annual_orders if annual_orders > 0 else 0.0

        annual_order_cost = annual_orders * s_cost
        annual_holding_cost = (eoq / 2.0) * unit_holding_cost
        total_inventory_cost = annual_order_cost + annual_holding_cost
        capital_in_cycle_stock = (eoq / 2.0) * unit_price

        return {
            "category": metrics["category"],
            "annual_demand_units": annual_demand,
            "unit_price": unit_price,
            "order_setup_cost": round(s_cost, 2),
            "holding_cost_rate": round(h_rate, 3),
            "unit_holding_cost_annual": round(unit_holding_cost, 2),
            "eoq_units": int(math.ceil(eoq)),
            "eoq_units_exact": round(eoq, 2),
            "annual_order_frequency": round(annual_orders, 2),
            "order_cycle_interval_days": round(cycle_days, 1),
            "annual_ordering_cost": round(annual_order_cost, 2),
            "annual_holding_cost": round(annual_holding_cost, 2),
            "total_annual_inventory_cost": round(total_inventory_cost, 2),
            "working_capital_in_cycle_stock": round(capital_in_cycle_stock, 2),
        }

    def simulate_stockout_risk(
        self,
        service_level: float = 0.95,
        lead_time_days: Optional[float] = None,
        lead_time_std: Optional[float] = None,
        category: Optional[str] = None,
        gross_margin_pct: float = 0.40,
    ) -> Dict[str, Any]:
        """
        Simulates stockout probabilities and financial margin at risk using the
        Standard Normal Loss Function G(k) = phi(k) - k * (1 - Phi(k)).
        """
        if not (0.01 < gross_margin_pct < 1.0):
            raise ValueError(f"Gross margin percentage must be in (0.01, 1.0), got {gross_margin_pct}")

        ss_res = self.calculate_safety_stock_and_rop(
            service_level=service_level,
            lead_time_days=lead_time_days,
            lead_time_std=lead_time_std,
            category=category,
        )
        eoq_res = self.calculate_eoq(category=category)

        k = ss_res["z_score"]
        ltd_std = ss_res["lead_time_demand_std"]
        eoq = eoq_res["eoq_units_exact"]
        n_orders = eoq_res["annual_order_frequency"]
        unit_price = eoq_res["unit_price"]

        # Standard normal unit loss integral
        norm_pdf = stats.norm.pdf(k)
        norm_cdf = stats.norm.cdf(k)
        g_k = norm_pdf - k * (1.0 - norm_cdf)

        expected_unmet_units_per_cycle = ltd_std * g_k
        annual_unmet_units = expected_unmet_units_per_cycle * n_orders
        annual_lost_revenue = annual_unmet_units * unit_price
        annual_lost_gross_margin = annual_lost_revenue * gross_margin_pct
        stockout_prob_per_cycle = 1.0 - service_level
        cycle_fill_rate = max(0.0, min(1.0, 1.0 - (expected_unmet_units_per_cycle / eoq if eoq > 0 else 0.0)))

        return {
            "category": ss_res["category"],
            "service_level": round(service_level, 4),
            "safety_factor_k": round(k, 3),
            "loss_function_gk": round(float(g_k), 5),
            "stockout_probability_per_cycle": round(stockout_prob_per_cycle, 4),
            "cycle_fill_rate_percentage": round(cycle_fill_rate * 100.0, 2),
            "expected_unmet_units_per_cycle": round(expected_unmet_units_per_cycle, 3),
            "annual_expected_unmet_units": round(annual_unmet_units, 2),
            "annual_lost_revenue": round(annual_lost_revenue, 2),
            "annual_lost_gross_margin": round(annual_lost_gross_margin, 2),
            "safety_stock_units": ss_res["safety_stock_units"],
            "reorder_point_units": ss_res["reorder_point_units"],
            "safety_stock_capital": ss_res["safety_stock_capital"],
            "gross_margin_pct": gross_margin_pct,
        }

    def perform_abc_fsn_analysis(self) -> Dict[str, Any]:
        """
        Executes multi-criteria ABC-FSN strategic inventory stratification:
        - ABC: Cumulative revenue impact (A <= 70%, B 70-90%, C > 90%)
        - FSN: Movement velocity (Fast, Slow, Non-moving based on daily sales frequency).
        """
        # Stratify across product categories and price tiers
        items = (
            self.df.groupby(["Product Category", "Price per Unit"])
            .agg(
                order_count=("Transaction ID", "count"),
                total_units=("Quantity", "sum"),
                total_revenue=("Total Amount", "sum"),
            )
            .reset_index()
            .sort_values(by="total_revenue", ascending=False)
        )

        total_rev = items["total_revenue"].sum()
        items["revenue_share"] = items["total_revenue"] / total_rev
        items["cum_revenue_share"] = items["revenue_share"].cumsum()

        # ABC Classification
        def classify_abc(cum_share: float) -> str:
            if cum_share <= 0.70:
                return "Class A (High Value)"
            elif cum_share <= 0.90:
                return "Class B (Moderate Value)"
            return "Class C (Low Value)"

        items["abc_class"] = items["cum_revenue_share"].apply(classify_abc)

        # FSN Classification based on order velocity
        freq_q33 = items["order_count"].quantile(0.33)
        freq_q66 = items["order_count"].quantile(0.66)

        def classify_fsn(count: int) -> str:
            if count >= freq_q66:
                return "Fast-Moving (F)"
            elif count >= freq_q33:
                return "Slow-Moving (S)"
            return "Non-Moving / Dormant (N)"

        items["fsn_class"] = items["order_count"].apply(classify_fsn)

        # Policy recommendation mapping
        policies = {
            ("Class A (High Value)", "Fast-Moving (F)"): "Strict continuous review, automated EDI replenishment, priority vendor SLA.",
            ("Class A (High Value)", "Slow-Moving (S)"): "High dollar impact. Lean safety stocks, JIT orders to avoid working capital lockup.",
            ("Class A (High Value)", "Non-Moving / Dormant (N)"): "Special order only. Zero safety stock, eliminate excess catalog bloat.",
            ("Class B (Moderate Value)", "Fast-Moving (F)"): "Standard periodic review (weekly), safety stock at 95% service level.",
            ("Class B (Moderate Value)", "Slow-Moving (S)"): "Bi-weekly review, EOQ batching, moderate buffer.",
            ("Class B (Moderate Value)", "Non-Moving / Dormant (N)"): "Markdown promotional clearance candidate to recover capital.",
            ("Class C (Low Value)", "Fast-Moving (F)"): "Bulk reordering, large batch size, low review overhead.",
            ("Class C (Low Value)", "Slow-Moving (S)"): "Infrequent bulk replenishment, low priority monitoring.",
            ("Class C (Low Value)", "Non-Moving / Dormant (N)"): "Phase-out or bundle bundle incentive with A-Fast products.",
        }

        stratified_records = []
        for _, row in items.iterrows():
            abc = row["abc_class"]
            fsn = row["fsn_class"]
            policy = policies.get((abc, fsn), "Standard replenishment guidelines.")
            stratified_records.append(
                {
                    "product_category": row["Product Category"],
                    "price_tier": float(row["Price per Unit"]),
                    "order_count": int(row["order_count"]),
                    "total_units": int(row["total_units"]),
                    "total_revenue": round(float(row["total_revenue"]), 2),
                    "revenue_share_pct": round(float(row["revenue_share"]) * 100.0, 2),
                    "cum_revenue_share_pct": round(float(row["cum_revenue_share"]) * 100.0, 2),
                    "abc_classification": abc,
                    "fsn_classification": fsn,
                    "joint_matrix_tier": f"{abc[:7]} / {fsn[:4]}",
                    "governance_policy": policy,
                }
            )

        # Summary counts
        abc_counts = items["abc_class"].value_counts().to_dict()
        fsn_counts = items["fsn_class"].value_counts().to_dict()

        return {
            "total_sku_tiers": len(items),
            "storewide_revenue": round(float(total_rev), 2),
            "abc_distribution": abc_counts,
            "fsn_distribution": fsn_counts,
            "sku_stratifications": stratified_records,
        }

    def get_storewide_inventory_summary(
        self, service_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Consolidates storewide and category-by-category supply chain parameters,
        working capital requirements, and total safety buffer investments.
        """
        categories = sorted(self.df["Product Category"].unique().tolist())
        cat_summaries = {}
        total_safety_capital = 0.0
        total_cycle_capital = 0.0
        total_annual_holding_cost = 0.0

        for cat in categories:
            ss = self.calculate_safety_stock_and_rop(service_level=service_level, category=cat)
            eoq = self.calculate_eoq(category=cat)
            risk = self.simulate_stockout_risk(service_level=service_level, category=cat)

            cat_summaries[cat] = {
                "daily_mean_units": ss["daily_mean_demand"],
                "daily_std_units": ss["daily_std_demand"],
                "safety_stock_units": ss["safety_stock_units"],
                "reorder_point_units": ss["reorder_point_units"],
                "safety_stock_capital": ss["safety_stock_capital"],
                "eoq_units": eoq["eoq_units"],
                "order_cycle_days": eoq["order_cycle_interval_days"],
                "working_capital_in_cycle_stock": eoq["working_capital_in_cycle_stock"],
                "annual_inventory_cost": eoq["total_annual_inventory_cost"],
                "expected_annual_lost_margin": risk["annual_lost_gross_margin"],
                "cycle_fill_rate_pct": risk["cycle_fill_rate_percentage"],
            }
            total_safety_capital += ss["safety_stock_capital"]
            total_cycle_capital += eoq["working_capital_in_cycle_stock"]
            total_annual_holding_cost += eoq["annual_holding_cost"]

        return {
            "target_service_level": round(service_level, 4),
            "total_working_capital_committed": round(total_safety_capital + total_cycle_capital, 2),
            "total_safety_stock_capital": round(total_safety_capital, 2),
            "total_cycle_stock_capital": round(total_cycle_capital, 2),
            "total_annual_holding_cost": round(total_annual_holding_cost, 2),
            "categories": cat_summaries,
        }
