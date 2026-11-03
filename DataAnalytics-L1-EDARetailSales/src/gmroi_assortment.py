"""
Gross Margin Return on Investment (GMROI) & Merchandise Assortment Rationalization Engine.

Provides:
1. Category and catalog price tier GMROI benchmarking:
   - Formula: GMROI = Gross Margin $ / Average Inventory Cost $ = Gross Margin % * Inventory Turns
2. Strategic Margin-Turn 4-Quadrant Assortment Matrix:
   - Stars: High Margin & High Turns (Core Profit Engines)
   - High-Yield Specialists: High Margin & Low Turns (Niche Premium)
   - Traffic Drivers: Low Margin & High Turns (Footfall & Basket Builders)
   - Dead Stock / Capital Traps: Low Margin & Low Turns (Clearance / Rationalization candidates)
3. Assortment Rationalization & SKU Productivity Diagnostics:
   - Evaluates capital drag, revenue contribution vs inventory burden, and cannibalization risk.
4. Executive Working Capital Reallocation Simulator:
   - Simulates shifting underperforming inventory capital into high-GMROI stars to maximize commercial yield.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.analytics import RetailAnalyticsEngine


class GMROIAssortmentEngine:
    """
    Enterprise retail merchandise economics and assortment rationalization engine.
    Calibrates GMROI, inventory turns, margin efficiency, and capital reallocation strategies.
    """

    # Industry benchmark gross margin rates across retail categories
    CATEGORY_GROSS_MARGINS = {
        "Beauty": 0.52,       # 52% Gross Margin
        "Clothing": 0.46,     # 46% Gross Margin
        "Electronics": 0.28,  # 28% Gross Margin
    }

    # Calibrated baseline annual inventory turns by category
    CATEGORY_BASE_TURNS = {
        "Beauty": 5.4,
        "Clothing": 4.8,
        "Electronics": 6.2,
    }

    def __init__(
        self,
        data_path: Optional[Path] = None,
        custom_margins: Optional[Dict[str, float]] = None,
    ):
        self.analytics = RetailAnalyticsEngine(data_path=data_path)
        self.df = self.analytics.df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.margins = custom_margins or self.CATEGORY_GROSS_MARGINS.copy()

    def calculate_category_gmroi(self) -> Dict[str, Any]:
        """
        Computes GMROI and capital efficiency metrics for all merchandise categories.
        """
        category_metrics = []
        store_revenue = float(self.df["Total Amount"].sum())
        store_gross_profit = 0.0
        store_avg_inventory_cost = 0.0

        for cat in sorted(self.df["Product Category"].unique()):
            cat_df = self.df[self.df["Product Category"] == cat]
            rev = float(cat_df["Total Amount"].sum())
            units = int(cat_df["Quantity"].sum())
            margin_pct = self.margins.get(cat, 0.40)
            turns = self.CATEGORY_BASE_TURNS.get(cat, 5.0)

            cogs = rev * (1.0 - margin_pct)
            gross_margin_dollars = rev * margin_pct
            avg_inventory_cost = cogs / turns if turns > 0 else 0.0
            gmroi = gross_margin_dollars / avg_inventory_cost if avg_inventory_cost > 0 else 0.0

            store_gross_profit += gross_margin_dollars
            store_avg_inventory_cost += avg_inventory_cost

            # Commercial rating
            if gmroi >= 2.5:
                rating = "Elite / High Capital Productivity"
            elif gmroi >= 1.8:
                rating = "Healthy / Above Benchmark"
            elif gmroi >= 1.2:
                rating = "Acceptable / Standard"
            else:
                rating = "Capital Dilutive / Underperforming"

            category_metrics.append({
                "category": cat,
                "annual_revenue": round(rev, 2),
                "units_sold": units,
                "revenue_share_pct": round(rev / store_revenue * 100, 2),
                "gross_margin_pct": round(margin_pct * 100, 2),
                "gross_margin_dollars": round(gross_margin_dollars, 2),
                "inventory_turns": round(turns, 2),
                "average_inventory_cost": round(avg_inventory_cost, 2),
                "gmroi_ratio": round(gmroi, 2),
                "commercial_rating": rating,
            })

        store_gmroi = store_gross_profit / store_avg_inventory_cost if store_avg_inventory_cost > 0 else 0.0
        store_blended_margin = (store_gross_profit / store_revenue) * 100 if store_revenue > 0 else 0.0

        return {
            "store_total_revenue": round(store_revenue, 2),
            "store_total_gross_profit": round(store_gross_profit, 2),
            "store_blended_gross_margin_pct": round(store_blended_margin, 2),
            "store_total_inventory_investment": round(store_avg_inventory_cost, 2),
            "store_blended_gmroi": round(store_gmroi, 2),
            "categories": category_metrics,
        }

    def compute_margin_turn_matrix(self) -> Dict[str, Any]:
        """
        Profiles catalog price tiers and sub-segments on the Margin-Turn 4-Quadrant Matrix:
        Stars, High-Yield Specialists, Traffic Drivers, Dead Stock.
        """
        records = []
        groups = self.df.groupby(["Product Category", "Price per Unit"])

        for (cat, price), group in groups:
            rev = float(group["Total Amount"].sum())
            units = int(group["Quantity"].sum())
            orders = len(group)
            margin_pct = self.margins.get(cat, 0.40)

            # High price units tend to have slightly lower turns; low price units have higher velocity
            base_turn = self.CATEGORY_BASE_TURNS.get(cat, 5.0)
            if price >= 300:
                tier_turn = base_turn * 0.75
            elif price <= 50:
                tier_turn = base_turn * 1.25
            else:
                tier_turn = base_turn

            cogs = rev * (1.0 - margin_pct)
            gross_margin_dollars = rev * margin_pct
            avg_inv_cost = cogs / tier_turn if tier_turn > 0 else 0.0
            gmroi = gross_margin_dollars / avg_inv_cost if avg_inv_cost > 0 else 0.0

            records.append({
                "sku_id": f"{cat}_{price}",
                "category": cat,
                "price_per_unit": float(price),
                "revenue": rev,
                "units": units,
                "orders": orders,
                "gross_margin_pct": margin_pct,
                "turns": tier_turn,
                "gross_margin_dollars": gross_margin_dollars,
                "average_inventory_cost": avg_inv_cost,
                "gmroi": gmroi,
            })

        sku_df = pd.DataFrame(records)
        median_margin = sku_df["gross_margin_pct"].median()
        median_turn = sku_df["turns"].median()

        classified_skus = []
        quadrant_summary = {
            "Stars": {"count": 0, "revenue": 0.0, "gross_profit": 0.0, "avg_gmroi": 0.0},
            "High-Yield Specialists": {"count": 0, "revenue": 0.0, "gross_profit": 0.0, "avg_gmroi": 0.0},
            "Traffic Drivers": {"count": 0, "revenue": 0.0, "gross_profit": 0.0, "avg_gmroi": 0.0},
            "Dead Stock / Capital Traps": {"count": 0, "revenue": 0.0, "gross_profit": 0.0, "avg_gmroi": 0.0},
        }

        for _, row in sku_df.iterrows():
            margin_val = row["gross_margin_pct"]
            turn_val = row["turns"]

            if margin_val >= median_margin and turn_val >= median_turn:
                quad = "Stars"
                action = "Expand Assortment Depth & Priority Display"
            elif margin_val >= median_margin and turn_val < median_turn:
                quad = "High-Yield Specialists"
                action = "Cross-Sell Bundling & Targeted Promotion"
            elif margin_val < median_margin and turn_val >= median_turn:
                quad = "Traffic Drivers"
                action = "Protect Margin Floor & Attach High-Margin SKUs"
            else:
                quad = "Dead Stock / Capital Traps"
                action = "Markdown Clearance / SKU Rationalization"

            quadrant_summary[quad]["count"] += 1
            quadrant_summary[quad]["revenue"] += row["revenue"]
            quadrant_summary[quad]["gross_profit"] += row["gross_margin_dollars"]

            classified_skus.append({
                "sku_id": row["sku_id"],
                "category": row["category"],
                "price_tier": row["price_per_unit"],
                "revenue": round(row["revenue"], 2),
                "units_sold": row["units"],
                "gross_margin_pct": round(row["gross_margin_pct"] * 100, 2),
                "inventory_turns": round(row["turns"], 2),
                "average_inventory_cost": round(row["average_inventory_cost"], 2),
                "gmroi": round(row["gmroi"], 2),
                "quadrant": quad,
                "merchandising_action": action,
            })

        for q in quadrant_summary:
            q_skus = [s for s in classified_skus if s["quadrant"] == q]
            if q_skus:
                quadrant_summary[q]["avg_gmroi"] = round(float(np.mean([s["gmroi"] for s in q_skus])), 2)
            quadrant_summary[q]["revenue"] = round(quadrant_summary[q]["revenue"], 2)
            quadrant_summary[q]["gross_profit"] = round(quadrant_summary[q]["gross_profit"], 2)

        return {
            "thresholds": {
                "median_gross_margin_pct": round(float(median_margin) * 100, 2),
                "median_inventory_turns": round(float(median_turn), 2),
            },
            "quadrant_summary": quadrant_summary,
            "classified_skus": classified_skus,
        }

    def evaluate_assortment_rationalization(self) -> Dict[str, Any]:
        """
        Diagnoses SKU assortment productivity, capital drag, and identifies
        candidates for rationalization, delisting, or markdown clearance.
        """
        matrix_data = self.compute_margin_turn_matrix()
        skus = matrix_data["classified_skus"]

        total_inv_cost = sum([s["average_inventory_cost"] for s in skus])
        total_rev = sum([s["revenue"] for s in skus])

        rationalization_candidates = []
        for s in skus:
            rev_share = (s["revenue"] / total_rev) * 100 if total_rev > 0 else 0.0
            cost_share = (s["average_inventory_cost"] / total_inv_cost) * 100 if total_inv_cost > 0 else 0.0
            capital_drag_ratio = round(cost_share / rev_share, 2) if rev_share > 0 else 999.0

            is_candidate = (s["quadrant"] == "Dead Stock / Capital Traps") or (capital_drag_ratio > 1.25)

            if is_candidate:
                rationalization_candidates.append({
                    "sku_id": s["sku_id"],
                    "category": s["category"],
                    "price_tier": s["price_tier"],
                    "quadrant": s["quadrant"],
                    "gmroi": s["gmroi"],
                    "revenue_share_pct": round(rev_share, 2),
                    "inventory_cost_share_pct": round(cost_share, 2),
                    "capital_drag_ratio": capital_drag_ratio,
                    "inventory_capital_tied_up": s["average_inventory_cost"],
                    "recommended_action": s["merchandising_action"],
                })

        # Sort candidates by highest capital tied up
        rationalization_candidates.sort(key=lambda x: x["inventory_capital_tied_up"], reverse=True)
        total_tied_capital = sum([c["inventory_capital_tied_up"] for c in rationalization_candidates])

        return {
            "total_skus_evaluated": len(skus),
            "rationalization_candidates_count": len(rationalization_candidates),
            "total_capital_at_risk": round(total_tied_capital, 2),
            "candidates": rationalization_candidates,
        }

    def simulate_working_capital_reallocation(
        self,
        reallocation_pct: float = 0.35,
        source_quadrant: str = "Dead Stock / Capital Traps",
        target_quadrant: str = "Stars",
    ) -> Dict[str, Any]:
        """
        Simulates the commercial impact of shifting inventory capital from underperforming
        SKUs/quadrants directly into high-yield Stars.

        Parameters:
        - reallocation_pct: Fraction of source inventory capital to reallocate (0.10 to 0.75).
        - source_quadrant: Quadrant to divest from.
        - target_quadrant: High-performing quadrant to reinvest into.
        """
        if not (0.05 <= reallocation_pct <= 0.90):
            raise ValueError(f"reallocation_pct must be between 0.05 and 0.90. Got {reallocation_pct}")

        matrix_data = self.compute_margin_turn_matrix()
        quads = matrix_data["quadrant_summary"]

        if source_quadrant not in quads:
            raise ValueError(f"Invalid source_quadrant '{source_quadrant}'. Must be one of {list(quads.keys())}")
        if target_quadrant not in quads:
            raise ValueError(f"Invalid target_quadrant '{target_quadrant}'. Must be one of {list(quads.keys())}")

        skus = matrix_data["classified_skus"]
        source_skus = [s for s in skus if s["quadrant"] == source_quadrant]
        target_skus = [s for s in skus if s["quadrant"] == target_quadrant]

        source_capital = sum([s["average_inventory_cost"] for s in source_skus])
        target_capital = sum([s["average_inventory_cost"] for s in target_skus])

        source_gmroi = float(np.mean([s["gmroi"] for s in source_skus])) if source_skus else 1.0
        target_gmroi = float(np.mean([s["gmroi"] for s in target_skus])) if target_skus else 2.5

        # Capital to reallocate
        reallocated_amount = source_capital * reallocation_pct

        # Lost gross profit from source vs gained gross profit from target
        lost_gross_profit = reallocated_amount * source_gmroi
        gained_gross_profit = reallocated_amount * target_gmroi
        net_profit_uplift = gained_gross_profit - lost_gross_profit

        # Baseline store metrics
        baseline_summary = self.calculate_category_gmroi()
        base_profit = baseline_summary["store_total_gross_profit"]
        base_inv = baseline_summary["store_total_inventory_investment"]
        base_gmroi = baseline_summary["store_blended_gmroi"]

        projected_profit = base_profit + net_profit_uplift
        projected_gmroi = projected_profit / base_inv if base_inv > 0 else 0.0

        return {
            "source_quadrant": source_quadrant,
            "target_quadrant": target_quadrant,
            "reallocation_pct": round(reallocation_pct * 100, 2),
            "capital_reallocated_dollars": round(reallocated_amount, 2),
            "source_capital_baseline": round(source_capital, 2),
            "target_capital_baseline": round(target_capital, 2),
            "source_gmroi": round(source_gmroi, 2),
            "target_gmroi": round(target_gmroi, 2),
            "incremental_gross_profit": round(net_profit_uplift, 2),
            "profit_uplift_pct": round((net_profit_uplift / base_profit * 100) if base_profit > 0 else 0.0, 2),
            "baseline_store_gmroi": round(base_gmroi, 2),
            "projected_store_gmroi": round(projected_gmroi, 2),
            "gmroi_delta": round(projected_gmroi - base_gmroi, 2),
        }
