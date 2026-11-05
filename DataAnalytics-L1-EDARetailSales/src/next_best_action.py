"""
Causal Uplift Modeling & Next-Best-Action (NBA) Decision Engine.

Provides:
1. Two-Model Causal Uplift & Incremental Treatment Effect (ITE) Modeling:
   - Partitions shoppers into classical 4 causal response quadrants:
     * Persuadables: Incremental buyers only if treated (Positive Treatment Effect tau > 0)
     * Sure Things: Buy regardless of treatment (Zero incremental lift, margin dilution risk)
     * Lost Causes: Inactive regardless of treatment (Zero lift, wasted spend)
     * Sleeping Dogs / Do Not Disturb: Negative reaction / churn triggered by contact (tau < 0)
2. Next-Best-Action (NBA) Multi-Channel Optimization Engine:
   - Evaluates action portfolio: VIP Concierge, Cross-Category Voucher, App Push Flash, Email Digest, No-Contact Holdout
   - Utility scoring combining causal uplift probability, expected gross margin, intervention cost, and fatigue penalties
3. Campaign Portfolio Optimization & Cannibalization Prevention Simulator:
   - Contrasts naive propensity targeting (which subsidizes Sure Things) vs Uplift Causal targeting
   - Quantifies avoided cannibalization dollars and net marketing campaign ROI
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.analytics import RetailAnalyticsEngine


class NextBestActionEngine:
    """
    Enterprise customer micro-targeting, causal uplift estimation,
    and Next-Best-Action (NBA) algorithmic dispatch engine.
    """

    # Action portfolio with unit intervention costs, channel fatigue weights, and typical conversion multipliers
    ACTION_PORTFOLIO: Dict[str, Dict[str, Any]] = {
        "VIP_CONCIERGE": {
            "name": "VIP Personal Concierge & Private Fitting",
            "channel": "White-Glove Call / In-Store",
            "unit_cost": 22.0,
            "fatigue_weight": 0.40,
            "base_lift": 0.28,
            "min_spend_hurdle": 400.0,
        },
        "CROSS_SELL_VOUCHER": {
            "name": "Cross-Category Margin-Protected Voucher",
            "channel": "SMS / Direct Mail",
            "unit_cost": 5.5,
            "fatigue_weight": 0.25,
            "base_lift": 0.18,
            "min_spend_hurdle": 100.0,
        },
        "APP_PUSH_FLASH": {
            "name": "Personalized App Push Flash Incentive",
            "channel": "Mobile Push Notification",
            "unit_cost": 0.15,
            "fatigue_weight": 0.10,
            "base_lift": 0.08,
            "min_spend_hurdle": 0.0,
        },
        "EMAIL_CURATED": {
            "name": "Curated Editorial Lookbook & Product Drop",
            "channel": "Email Digest",
            "unit_cost": 0.05,
            "fatigue_weight": 0.05,
            "base_lift": 0.05,
            "min_spend_hurdle": 0.0,
        },
        "ORGANIC_HOLDOUT": {
            "name": "Organic Holdout (No Contact)",
            "channel": "None (Preserve Margin)",
            "unit_cost": 0.0,
            "fatigue_weight": 0.0,
            "base_lift": 0.0,
            "min_spend_hurdle": 0.0,
        },
    }

    def __init__(self, data_path: Optional[Path] = None, seed: int = 42):
        self.analytics = RetailAnalyticsEngine(data_path=data_path)
        self.df = self.analytics.df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.rng = np.random.default_rng(seed)
        self._prepare_customer_features()

    def _prepare_customer_features(self) -> None:
        """
        Extracts aggregate behavioral customer features for causal uplift modeling.
        """
        cust_group = self.df.groupby("Customer ID").agg(
            total_spend=("Total Amount", "sum"),
            transaction_count=("Transaction ID", "count"),
            avg_order_value=("Total Amount", "mean"),
            total_quantity=("Quantity", "sum"),
            primary_category=("Product Category", lambda x: x.mode()[0] if not x.empty else "Clothing"),
            gender=("Gender", "first"),
            age=("Age", "first"),
            last_date=("Date", "max"),
        ).reset_index()

        max_date = self.df["Date"].max()
        cust_group["recency_days"] = (max_date - cust_group["last_date"]).dt.days

        # Synthesize baseline conversion probability P(Y=1|X, T=0)
        # Higher spenders and more frequent buyers have higher baseline organic probability
        spend_norm = np.clip(cust_group["total_spend"] / 1500.0, 0.0, 1.0)
        recency_penalty = np.clip(cust_group["recency_days"] / 365.0, 0.0, 1.0) * 0.4
        freq_bonus = np.clip(cust_group["transaction_count"] / 5.0, 0.0, 0.3)

        base_prob = 0.15 + (0.50 * spend_norm) + freq_bonus - recency_penalty
        cust_group["organic_conversion_prob"] = np.clip(base_prob, 0.05, 0.90)

        # Causal Uplift (Individual Treatment Effect tau = P(Y=1|T=1) - P(Y=1|T=0))
        # Persuadables typically have mid-level baseline propensity (not too high, not too low)
        # Moderate spenders, active within 60-180 days have the highest uplift sensitivity
        mid_propensity_factor = 1.0 - (2.0 * np.abs(cust_group["organic_conversion_prob"] - 0.45))
        mid_propensity_factor = np.clip(mid_propensity_factor, -0.2, 0.8)

        # Sleeping dog penalty for customers with very low recency who hate frequent contact
        churn_risk_penalty = np.where(cust_group["recency_days"] > 300, -0.15, 0.0)

        ite_tau = (0.22 * mid_propensity_factor) + churn_risk_penalty
        # Add small deterministic variation based on age and category
        cat_bonus = cust_group["primary_category"].map(
            {"Beauty": 0.04, "Clothing": 0.02, "Electronics": -0.01}
        ).fillna(0.0)
        ite_tau = np.clip(ite_tau + cat_bonus, -0.15, 0.45)

        cust_group["ite_uplift"] = np.round(ite_tau, 4)
        cust_group["treated_conversion_prob"] = np.clip(
            cust_group["organic_conversion_prob"] + cust_group["ite_uplift"],
            0.02,
            0.98,
        )

        # Causal Quadrant Classification
        quadrants = []
        for _, row in cust_group.iterrows():
            tau = row["ite_uplift"]
            p0 = row["organic_conversion_prob"]
            if tau < -0.02:
                quadrants.append("Sleeping Dogs / Do Not Disturb")
            elif p0 >= 0.55 and tau < 0.08:
                quadrants.append("Sure Things")
            elif tau >= 0.08:
                quadrants.append("Persuadables")
            else:
                quadrants.append("Lost Causes")

        cust_group["causal_quadrant"] = quadrants
        self.cust_df = cust_group

    def get_uplift_summary(self) -> Dict[str, Any]:
        """
        Summarizes causal uplift distribution across the customer population.
        """
        total_customers = len(self.cust_df)
        quadrant_counts = self.cust_df["causal_quadrant"].value_counts().to_dict()

        quadrant_profiles = []
        for q_name in [
            "Persuadables",
            "Sure Things",
            "Lost Causes",
            "Sleeping Dogs / Do Not Disturb",
        ]:
            count = int(quadrant_counts.get(q_name, 0))
            pct = (count / total_customers) * 100.0 if total_customers > 0 else 0.0
            subset = self.cust_df[self.cust_df["causal_quadrant"] == q_name]
            avg_spend = float(subset["total_spend"].mean()) if count > 0 else 0.0
            avg_tau = float(subset["ite_uplift"].mean()) if count > 0 else 0.0
            avg_p0 = float(subset["organic_conversion_prob"].mean()) if count > 0 else 0.0

            if q_name == "Persuadables":
                strategy = "Prime Marketing Target: Allocate promotional vouchers and personal outreach."
            elif q_name == "Sure Things":
                strategy = "Margin Protection: Suppress discounts; engage with non-monetary brand content."
            elif q_name == "Lost Causes":
                strategy = "Spend Minimization: Suppress high-cost channels; low-frequency automated email only."
            else:
                strategy = "Strict Exclusion: Suppress all promotional touchpoints to prevent unsubscribes/churn."

            quadrant_profiles.append({
                "quadrant": q_name,
                "customer_count": count,
                "population_share_pct": round(pct, 2),
                "avg_historical_spend": round(avg_spend, 2),
                "avg_baseline_conversion_prob": round(avg_p0, 4),
                "avg_incremental_uplift_tau": round(avg_tau, 4),
                "strategic_recommendation": strategy,
            })

        avg_overall_tau = float(self.cust_df["ite_uplift"].mean())
        persuadable_count = int(quadrant_counts.get("Persuadables", 0))

        return {
            "total_customers": total_customers,
            "overall_mean_uplift_tau": round(avg_overall_tau, 4),
            "persuadable_population": persuadable_count,
            "persuadable_share_pct": round((persuadable_count / total_customers) * 100.0, 2),
            "quadrant_breakdown": quadrant_profiles,
        }

    def get_customer_uplift_profile(self, customer_id: str) -> Dict[str, Any]:
        """
        Retrieves individualized causal uplift profile for a specific customer.
        """
        match = self.cust_df[self.cust_df["Customer ID"] == customer_id]
        if match.empty:
            raise ValueError(f"Customer '{customer_id}' not found in portfolio.")

        row = match.iloc[0]
        return {
            "customer_id": customer_id,
            "causal_quadrant": str(row["causal_quadrant"]),
            "individual_treatment_effect_tau": float(row["ite_uplift"]),
            "organic_conversion_prob": float(round(row["organic_conversion_prob"], 4)),
            "treated_conversion_prob": float(round(row["treated_conversion_prob"], 4)),
            "total_spend": float(round(row["total_spend"], 2)),
            "avg_order_value": float(round(row["avg_order_value"], 2)),
            "primary_category": str(row["primary_category"]),
            "recency_days": int(row["recency_days"]),
            "action_recommendation": (
                "Deploy Incentive Touchpoint"
                if row["causal_quadrant"] == "Persuadables"
                else "Suppress Financial Promotion"
            ),
        }

    def recommend_next_best_action(
        self,
        customer_id: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        category: Optional[str] = None,
        estimated_aov: float = 250.0,
        gross_margin_pct: float = 0.45,
        recent_touches_count: int = 1,
    ) -> Dict[str, Any]:
        """
        Selects the optimal Next-Best-Action (NBA) by maximizing risk-adjusted net expected utility.

        Utility = (Incremental Uplift Probability * Expected Gross Margin $) - Intervention Cost - Fatigue Penalty
        """
        if estimated_aov <= 0:
            raise ValueError("estimated_aov must be positive.")
        if not (0.05 <= gross_margin_pct <= 0.95):
            raise ValueError("gross_margin_pct must be between 0.05 and 0.95.")
        if recent_touches_count < 0:
            raise ValueError("recent_touches_count cannot be negative.")

        if customer_id and customer_id in self.cust_df["Customer ID"].values:
            cust_info = self.get_customer_uplift_profile(customer_id)
            tau = cust_info["individual_treatment_effect_tau"]
            quadrant = cust_info["causal_quadrant"]
            hist_spend = cust_info["total_spend"]
            category = cust_info["primary_category"]
        else:
            # Synthetic cold-start or cohort profiling
            cust_info = None
            category = category or "Clothing"
            hist_spend = estimated_aov * 2.0
            # Heuristic tau
            tau = 0.14
            quadrant = "Persuadables"

        expected_order_margin = estimated_aov * gross_margin_pct

        scored_actions = []
        for action_key, spec in self.ACTION_PORTFOLIO.items():
            cost = spec["unit_cost"]
            fatigue_wt = spec["fatigue_weight"]
            base_lift = spec["base_lift"]

            # Severe penalty if Sleeping Dogs or Sure Things get promotional discounts
            if quadrant == "Sleeping Dogs / Do Not Disturb" and action_key != "ORGANIC_HOLDOUT":
                net_utility = -50.0
                justification = "Action suppressed: Customer is at high churn risk from promotional contact."
                eligible = False
            elif quadrant == "Sure Things" and action_key in ["VIP_CONCIERGE", "CROSS_SELL_VOUCHER"]:
                # Margin cannibalization: customer would buy anyway, giving discount is pure loss
                cannibalization_loss = expected_order_margin * 0.15
                net_utility = -cannibalization_loss - cost
                justification = "Action suppressed: Customer is a Sure Thing; incentive causes margin dilution."
                eligible = False
            elif quadrant == "Lost Causes" and cost > 2.0:
                net_utility = -cost
                justification = "Action suppressed: Customer unresponsive to high-touch interventions."
                eligible = False
            elif hist_spend < spec["min_spend_hurdle"] and action_key == "VIP_CONCIERGE":
                net_utility = -10.0
                justification = f"Action suppressed: Customer spend below ${spec['min_spend_hurdle']} concierge threshold."
                eligible = False
            else:
                # Effective uplift = base tau scaled by action potency
                effective_tau = max(0.0, tau * (base_lift / 0.15))
                incremental_gross_profit = effective_tau * expected_order_margin
                fatigue_penalty = recent_touches_count * (fatigue_wt * 6.0)
                net_utility = incremental_gross_profit - cost - fatigue_penalty
                justification = (
                    f"Net expected gain +${net_utility:.2f} (Incremental Profit: +${incremental_gross_profit:.2f}, "
                    f"Cost: ${cost:.2f}, Fatigue: -${fatigue_penalty:.2f})"
                )
                eligible = True

            scored_actions.append({
                "action_key": action_key,
                "action_name": spec["name"],
                "channel": spec["channel"],
                "unit_cost": cost,
                "net_expected_utility": round(net_utility, 2),
                "is_eligible": eligible,
                "evaluation_rationale": justification,
            })

        # Rank actions by net expected utility
        scored_actions.sort(key=lambda x: x["net_expected_utility"], reverse=True)
        top_action = scored_actions[0]

        return {
            "customer_id": customer_id or "COHORT_SIMULATED",
            "causal_quadrant": quadrant,
            "individual_treatment_effect_tau": round(tau, 4),
            "expected_order_value": round(estimated_aov, 2),
            "expected_order_margin_dollars": round(expected_order_margin, 2),
            "recommended_action": top_action["action_name"],
            "recommended_channel": top_action["channel"],
            "top_action_key": top_action["action_key"],
            "projected_net_utility": top_action["net_expected_utility"],
            "action_portfolio_ranking": scored_actions,
        }

    def optimize_campaign_budget(
        self,
        campaign_budget: float = 10000.0,
        incentive_cost_per_touch: float = 8.0,
        avg_basket_value: float = 350.0,
        gross_margin_pct: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Simulates and benchmarks 3 marketing targeting strategies:
        1. Mass Marketing (Blast entire customer base)
        2. Naive Propensity Targeting (Target high spenders / Sure Things)
        3. Causal Uplift Targeting (Target Persuadables only, suppressing Sure Things & Sleeping Dogs)
        """
        if campaign_budget <= 0:
            raise ValueError("campaign_budget must be positive.")
        if incentive_cost_per_touch <= 0:
            raise ValueError("incentive_cost_per_touch must be positive.")
        if avg_basket_value <= 0:
            raise ValueError("avg_basket_value must be positive.")
        if not (0.1 <= gross_margin_pct <= 0.9):
            raise ValueError("gross_margin_pct must be between 0.1 and 0.9.")

        max_touches = int(campaign_budget // incentive_cost_per_touch)
        gross_margin_per_order = avg_basket_value * gross_margin_pct

        total_customers = len(self.cust_df)
        persuadables = self.cust_df[self.cust_df["causal_quadrant"] == "Persuadables"]
        sure_things = self.cust_df[self.cust_df["causal_quadrant"] == "Sure Things"]

        # Strategy 1: Mass Random Marketing
        mass_touches = min(total_customers, max_touches)
        mass_spend = mass_touches * incentive_cost_per_touch
        # Weighted mean tau across population
        mean_tau_pop = max(0.01, float(self.cust_df["ite_uplift"].mean()))
        mass_incremental_buyers = mass_touches * mean_tau_pop
        mass_gross_profit = mass_incremental_buyers * gross_margin_per_order
        mass_net_profit = mass_gross_profit - mass_spend
        mass_roi = (mass_net_profit / mass_spend) * 100.0 if mass_spend > 0 else 0.0

        # Strategy 2: Naive Propensity Targeting (Top spenders = Sure Things)
        # Select top spenders
        top_propensity = self.cust_df.sort_values(by="total_spend", ascending=False).head(max_touches)
        prop_touches = len(top_propensity)
        prop_spend = prop_touches * incentive_cost_per_touch
        prop_mean_tau = max(0.01, float(top_propensity["ite_uplift"].mean()))
        prop_incremental_buyers = prop_touches * prop_mean_tau
        prop_gross_profit = prop_incremental_buyers * gross_margin_per_order
        # Cannibalization cost: 15% discount given to customers who were buying anyway
        cannibalized_orders = len(top_propensity[top_propensity["causal_quadrant"] == "Sure Things"])
        cannibalized_margin_loss = cannibalized_orders * (avg_basket_value * 0.15)
        prop_net_profit = prop_gross_profit - prop_spend - cannibalized_margin_loss
        prop_roi = (prop_net_profit / prop_spend) * 100.0 if prop_spend > 0 else 0.0

        # Strategy 3: Causal Uplift Targeting (Rank by ITE tau)
        top_uplift = self.cust_df.sort_values(by="ite_uplift", ascending=False).head(max_touches)
        # Filter out sleeping dogs and pure sure things
        top_uplift_filtered = top_uplift[top_uplift["ite_uplift"] > 0.05]
        uplift_touches = len(top_uplift_filtered)
        uplift_spend = uplift_touches * incentive_cost_per_touch
        uplift_mean_tau = float(top_uplift_filtered["ite_uplift"].mean()) if uplift_touches > 0 else 0.20
        uplift_incremental_buyers = uplift_touches * uplift_mean_tau
        uplift_gross_profit = uplift_incremental_buyers * gross_margin_per_order
        uplift_cannibalization_saved = cannibalized_margin_loss  # Saved by suppressing Sure Things
        uplift_net_profit = uplift_gross_profit - uplift_spend
        uplift_roi = (uplift_net_profit / uplift_spend) * 100.0 if uplift_spend > 0 else 0.0

        return {
            "campaign_budget_dollars": round(campaign_budget, 2),
            "incentive_cost_per_touch": round(incentive_cost_per_touch, 2),
            "max_reach_capacity": max_touches,
            "strategies": {
                "mass_marketing": {
                    "strategy_name": "Mass Marketing (Untargeted Blast)",
                    "contacts_reached": mass_touches,
                    "budget_utilized": round(mass_spend, 2),
                    "incremental_buyers": round(mass_incremental_buyers, 1),
                    "gross_margin_generated": round(mass_gross_profit, 2),
                    "net_profit": round(mass_net_profit, 2),
                    "marketing_roi_pct": round(mass_roi, 2),
                },
                "propensity_targeting": {
                    "strategy_name": "Naive Propensity Targeting (Highest Spenders)",
                    "contacts_reached": prop_touches,
                    "budget_utilized": round(prop_spend, 2),
                    "incremental_buyers": round(prop_incremental_buyers, 1),
                    "cannibalization_deadweight_loss": round(cannibalized_margin_loss, 2),
                    "gross_margin_generated": round(prop_gross_profit, 2),
                    "net_profit": round(prop_net_profit, 2),
                    "marketing_roi_pct": round(prop_roi, 2),
                },
                "causal_uplift_targeting": {
                    "strategy_name": "Causal Uplift Targeting (Persuadables Only)",
                    "contacts_reached": uplift_touches,
                    "budget_utilized": round(uplift_spend, 2),
                    "incremental_buyers": round(uplift_incremental_buyers, 1),
                    "cannibalization_dollars_saved": round(uplift_cannibalization_saved, 2),
                    "gross_margin_generated": round(uplift_gross_profit, 2),
                    "net_profit": round(uplift_net_profit, 2),
                    "marketing_roi_pct": round(uplift_roi, 2),
                },
            },
            "uplift_advantage_vs_propensity": {
                "incremental_profit_delta": round(uplift_net_profit - prop_net_profit, 2),
                "roi_point_spread": round(uplift_roi - prop_roi, 2),
            },
        }
