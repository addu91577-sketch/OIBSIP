"""
Dynamic Real-Time Pricing & Multi-Armed Bandit (MAB) Rate Optimization Engine.

Provides:
1. Multi-Armed Bandit (MAB) pricing algorithms:
   - Thompson Sampling (Beta-Bernoulli conjugate distributions for conversion yield)
   - Upper Confidence Bound 1 (UCB1) with logarithmic regret bounds
   - Epsilon-Greedy with exponential exploration decay
2. Real-Time Dynamic Pricing Quote Generator:
   - Scarcity multipliers based on inventory stock-to-safety-stock ratios
   - Demand surge factor calibrated on rolling sales velocity
   - Strict category price floors (margin protection) and ceilings (churn prevention)
3. Offline & Online Regret Minimization Simulator:
   - Tracks cumulative regret against theoretical Oracle optimal pricing
   - Quantifies commercial revenue lift and margin preservation
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.analytics import RetailAnalyticsEngine


class DynamicPricingEngine:
    """
    Enterprise real-time dynamic pricing engine driven by Bayesian Multi-Armed Bandits,
    demand-velocity surges, inventory scarcity multipliers, and competitive constraints.
    """

    # Candidate pricing arms per category (Discounts, Parity, Premium tiers)
    CATEGORY_PRICE_ARMS: Dict[str, List[float]] = {
        "Beauty": [25.0, 30.0, 40.0, 50.0, 60.0],
        "Clothing": [25.0, 35.0, 50.0, 75.0, 100.0],
        "Electronics": [200.0, 250.0, 300.0, 400.0, 500.0],
    }

    # Baseline willingness-to-pay logistic parameters (intercept alpha, price sensitivity beta)
    DEMAND_CURVE_PARAMS: Dict[str, Dict[str, float]] = {
        "Beauty": {"alpha": 2.8, "beta": 0.055},
        "Clothing": {"alpha": 2.6, "beta": 0.045},
        "Electronics": {"alpha": 3.2, "beta": 0.0085},
    }

    # Gross margin baseline cost per unit
    UNIT_COST_BASELINES: Dict[str, float] = {
        "Beauty": 22.0,
        "Clothing": 24.0,
        "Electronics": 210.0,
    }

    def __init__(self, data_path: Optional[Path] = None, seed: int = 42):
        self.analytics = RetailAnalyticsEngine(data_path=data_path)
        self.df = self.analytics.df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.rng = np.random.default_rng(seed)

    def _conversion_probability(self, category: str, price: float) -> float:
        """
        Calculates ground-truth customer conversion probability using empirical logistic demand curve.
        """
        params = self.DEMAND_CURVE_PARAMS.get(
            category, {"alpha": 2.5, "beta": 0.02}
        )
        logits = params["alpha"] - (params["beta"] * price)
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -10.0, 10.0)))
        return float(np.clip(prob, 0.02, 0.98))

    def get_candidate_arms(self) -> Dict[str, Any]:
        """
        Returns catalog candidate price arms, baseline conversion estimates,
        and expected revenue yields per arm.
        """
        catalog_arms = {}
        for cat, arms in self.CATEGORY_PRICE_ARMS.items():
            cost = self.UNIT_COST_BASELINES[cat]
            arm_details = []
            for arm in arms:
                conv = self._conversion_probability(cat, arm)
                expected_rev = arm * conv
                expected_margin = (arm - cost) * conv if arm > cost else 0.0
                arm_details.append({
                    "price_arm": arm,
                    "conversion_probability": round(conv, 4),
                    "expected_revenue_per_impression": round(expected_rev, 2),
                    "expected_margin_per_impression": round(expected_margin, 2),
                    "unit_cost": cost,
                })
            # Find oracle arm (highest expected revenue)
            best_arm = max(arm_details, key=lambda x: x["expected_revenue_per_impression"])
            catalog_arms[cat] = {
                "arms": arm_details,
                "oracle_optimal_price": best_arm["price_arm"],
                "oracle_expected_revenue": best_arm["expected_revenue_per_impression"],
            }

        return {
            "categories": catalog_arms,
            "algorithms_supported": ["thompson_sampling", "ucb1", "epsilon_greedy"],
        }

    def simulate_bandit_learning(
        self,
        category: str = "Electronics",
        algorithm: str = "thompson_sampling",
        n_rounds: int = 1000,
        epsilon: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Simulates online Multi-Armed Bandit pricing exploration and regret minimization.

        Parameters:
            category: Target merchandise category ('Beauty', 'Clothing', 'Electronics')
            algorithm: 'thompson_sampling', 'ucb1', or 'epsilon_greedy'
            n_rounds: Number of sequential customer shopping sessions
            epsilon: Exploration parameter for epsilon-greedy

        Returns:
            Dictionary containing round-by-round trajectory, arm selection counts,
            cumulative revenue, and regret bounds.
        """
        if category not in self.CATEGORY_PRICE_ARMS:
            raise ValueError(
                f"Invalid category '{category}'. Available: {list(self.CATEGORY_PRICE_ARMS.keys())}"
            )
        if algorithm not in ["thompson_sampling", "ucb1", "epsilon_greedy"]:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. Must be 'thompson_sampling', 'ucb1', or 'epsilon_greedy'."
            )
        if n_rounds < 10 or n_rounds > 50000:
            raise ValueError("n_rounds must be between 10 and 50,000.")

        arms = self.CATEGORY_PRICE_ARMS[category]
        k = len(arms)
        cost = self.UNIT_COST_BASELINES[category]

        # Precompute oracle
        oracle_revenues = [arm * self._conversion_probability(category, arm) for arm in arms]
        best_arm_idx = int(np.argmax(oracle_revenues))
        oracle_expected_rev = oracle_revenues[best_arm_idx]

        # Bandit State Trackers
        counts = np.zeros(k, dtype=int)
        successes = np.zeros(k, dtype=int)  # conversions
        total_rewards = np.zeros(k, dtype=float)  # revenue collected

        # Prior for Thompson Sampling (Beta(1, 1))
        alpha_priors = np.ones(k, dtype=float)
        beta_priors = np.ones(k, dtype=float)

        history_rounds: List[int] = []
        cumulative_bandit_rev = 0.0
        cumulative_oracle_rev = 0.0
        cumulative_static_rev = 0.0
        # Baseline static pricing: middle tier arm
        static_arm_idx = k // 2
        static_price = arms[static_arm_idx]

        cumulative_regret = 0.0
        history_bandit_revenue: List[float] = []
        history_oracle_revenue: List[float] = []
        history_static_revenue: List[float] = []
        history_regret: List[float] = []
        arm_selected_log: List[int] = []

        sample_step = max(1, n_rounds // 100)

        for t in range(1, n_rounds + 1):
            # Arm Selection Policy
            if algorithm == "thompson_sampling":
                sampled_probs = [
                    self.rng.beta(alpha_priors[i], beta_priors[i]) for i in range(k)
                ]
                # Reward is expected revenue = price * sampled conversion prob
                sampled_expected_rev = [arms[i] * sampled_probs[i] for i in range(k)]
                chosen_idx = int(np.argmax(sampled_expected_rev))

            elif algorithm == "ucb1":
                # Exploration bonus with c=1.5
                if t <= k:
                    chosen_idx = t - 1
                else:
                    ucb_values = np.zeros(k)
                    for i in range(k):
                        avg_reward = total_rewards[i] / counts[i] if counts[i] > 0 else 0.0
                        bonus = 1.5 * np.sqrt((2.0 * np.log(t)) / counts[i]) if counts[i] > 0 else 1e5
                        ucb_values[i] = avg_reward + (arms[i] * bonus)
                    chosen_idx = int(np.argmax(ucb_values))

            else:  # epsilon_greedy with decaying epsilon
                decayed_eps = max(0.02, epsilon * (0.999 ** t))
                if self.rng.random() < decayed_eps:
                    chosen_idx = int(self.rng.integers(0, k))
                else:
                    avg_rewards = [
                        (total_rewards[i] / counts[i]) if counts[i] > 0 else 0.0
                        for i in range(k)
                    ]
                    chosen_idx = int(np.argmax(avg_rewards))

            chosen_price = arms[chosen_idx]
            arm_selected_log.append(chosen_idx)

            # Customer Environment Stochastic Response
            p_conv = self._conversion_probability(category, chosen_price)
            converted = bool(self.rng.random() < p_conv)
            revenue_realized = chosen_price if converted else 0.0

            # Static benchmark customer response
            static_conv = self._conversion_probability(category, static_price)
            static_converted = bool(self.rng.random() < static_conv)
            static_rev_realized = static_price if static_converted else 0.0

            # Update Bandit Trackers
            counts[chosen_idx] += 1
            if converted:
                successes[chosen_idx] += 1
            total_rewards[chosen_idx] += revenue_realized

            if algorithm == "thompson_sampling":
                if converted:
                    alpha_priors[chosen_idx] += 1.0
                else:
                    beta_priors[chosen_idx] += 1.0

            cumulative_bandit_rev += revenue_realized
            cumulative_oracle_rev += oracle_expected_rev
            cumulative_static_rev += static_rev_realized

            # Regret = Oracle expected reward - Chosen arm expected reward
            arm_expected = chosen_price * p_conv
            cumulative_regret += (oracle_expected_rev - arm_expected)

            if t % sample_step == 0 or t == n_rounds:
                history_rounds.append(t)
                history_bandit_revenue.append(round(cumulative_bandit_rev, 2))
                history_oracle_revenue.append(round(cumulative_oracle_rev, 2))
                history_static_revenue.append(round(cumulative_static_rev, 2))
                history_regret.append(round(cumulative_regret, 2))

        # Compile arm statistics
        arm_stats = []
        for i in range(k):
            arm_p = arms[i]
            arm_cnt = int(counts[i])
            arm_succ = int(successes[i])
            emp_conv = arm_succ / arm_cnt if arm_cnt > 0 else 0.0
            emp_rev = float(total_rewards[i])
            emp_margin = float((arm_p - cost) * arm_succ)
            arm_stats.append({
                "price_arm": arm_p,
                "pull_count": arm_cnt,
                "allocation_pct": round((arm_cnt / n_rounds) * 100.0, 2),
                "conversions": arm_succ,
                "empirical_conversion_rate": round(emp_conv, 4),
                "total_revenue_collected": round(emp_rev, 2),
                "total_gross_profit": round(emp_margin, 2),
                "is_oracle_optimal": bool(i == best_arm_idx),
            })

        rev_uplift_vs_static = cumulative_bandit_rev - cumulative_static_rev
        uplift_pct = (
            (rev_uplift_vs_static / cumulative_static_rev) * 100.0
            if cumulative_static_rev > 0
            else 0.0
        )

        return {
            "category": category,
            "algorithm": algorithm,
            "total_rounds": n_rounds,
            "oracle_optimal_arm": arms[best_arm_idx],
            "static_baseline_price": static_price,
            "cumulative_bandit_revenue": round(cumulative_bandit_rev, 2),
            "cumulative_static_revenue": round(cumulative_static_rev, 2),
            "cumulative_oracle_revenue": round(cumulative_oracle_rev, 2),
            "revenue_uplift_vs_static": round(rev_uplift_vs_static, 2),
            "revenue_uplift_pct": round(uplift_pct, 2),
            "total_regret": round(cumulative_regret, 2),
            "regret_per_round": round(cumulative_regret / n_rounds, 4),
            "arm_statistics": arm_stats,
            "learning_curve": {
                "rounds": history_rounds,
                "bandit_revenue": history_bandit_revenue,
                "static_revenue": history_static_revenue,
                "oracle_revenue": history_oracle_revenue,
                "cumulative_regret": history_regret,
            },
        }

    def calculate_dynamic_quote(
        self,
        category: str,
        base_price: float,
        current_stock: int,
        safety_stock: int = 25,
        demand_velocity_ratio: float = 1.0,
        competitor_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Computes real-time dynamic pricing quote integrating demand velocity,
        inventory scarcity multipliers, and competitive boundaries.

        Parameters:
            category: Product category ('Beauty', 'Clothing', 'Electronics')
            base_price: Catalog standard list price
            current_stock: Real-time on-hand stock count
            safety_stock: Calibrated safety stock threshold
            demand_velocity_ratio: Ratio of current sales velocity to 30-day baseline (e.g. 1.25 = +25% surge)
            competitor_price: Optional market competitor benchmark price
        """
        if category not in self.CATEGORY_PRICE_ARMS:
            raise ValueError(
                f"Invalid category '{category}'. Available: {list(self.CATEGORY_PRICE_ARMS.keys())}"
            )
        if base_price <= 0:
            raise ValueError("base_price must be strictly positive.")
        if current_stock < 0:
            raise ValueError("current_stock cannot be negative.")
        if safety_stock <= 0:
            raise ValueError("safety_stock must be strictly positive.")
        if demand_velocity_ratio <= 0:
            raise ValueError("demand_velocity_ratio must be strictly positive.")

        unit_cost = self.UNIT_COST_BASELINES.get(category, base_price * 0.5)

        # 1. Scarcity Multiplier (Surges when stock drops below safety stock)
        if current_stock == 0:
            scarcity_multiplier = 1.25  # High scarcity flag / waitlist premium
            stock_status = "Stockout Risk / Critical Scarcity"
        elif current_stock < safety_stock:
            deficit_pct = (safety_stock - current_stock) / safety_stock
            scarcity_multiplier = 1.0 + (0.20 * deficit_pct)
            stock_status = "Depleted / Sub-Safety-Stock"
        elif current_stock > safety_stock * 3.0:
            surplus_pct = min(1.0, (current_stock - safety_stock * 3.0) / (safety_stock * 3.0))
            scarcity_multiplier = 1.0 - (0.12 * surplus_pct)
            stock_status = "Surplus Inventory / Markdown Pressure"
        else:
            scarcity_multiplier = 1.0
            stock_status = "Optimal Stock Level"

        # 2. Demand Velocity Surge Multiplier
        # Normalize: 1.0 is neutral. If velocity > 1.0, apply gentle surge; if < 1.0, apply discount
        velocity_delta = demand_velocity_ratio - 1.0
        surge_multiplier = 1.0 + float(np.clip(velocity_delta * 0.18, -0.20, 0.25))

        # 3. Raw Composite Dynamic Price
        raw_price = base_price * scarcity_multiplier * surge_multiplier

        # 4. Competitor Clamping (if available)
        competitor_constrained = False
        if competitor_price is not None and competitor_price > 0:
            # Prevent pricing more than 15% above competitor unless extreme scarcity
            comp_ceiling = competitor_price * 1.15
            comp_floor = competitor_price * 0.85
            if raw_price > comp_ceiling and current_stock >= safety_stock:
                raw_price = comp_ceiling
                competitor_constrained = True
            elif raw_price < comp_floor:
                raw_price = comp_floor
                competitor_constrained = True

        # 5. Enterprise Guardrails: Floor & Ceiling
        # Floor: at least 15% margin above unit cost or 70% of base price
        price_floor = max(unit_cost * 1.15, base_price * 0.70)
        # Ceiling: maximum 140% of base price
        price_ceiling = base_price * 1.40

        clamped_price = float(np.clip(raw_price, price_floor, price_ceiling))
        rounded_dynamic_price = round(clamped_price, 2)

        # Expected conversion & margin analysis
        conv_prob = self._conversion_probability(category, rounded_dynamic_price)
        expected_margin_pct = (
            ((rounded_dynamic_price - unit_cost) / rounded_dynamic_price) * 100.0
            if rounded_dynamic_price > 0
            else 0.0
        )
        delta_vs_base_pct = ((rounded_dynamic_price - base_price) / base_price) * 100.0

        if delta_vs_base_pct > 2.0:
            strategy_action = "Surge / Scarcity Premium Applied"
        elif delta_vs_base_pct < -2.0:
            strategy_action = "Demand Clearance / Inventory Stimulation Discount"
        else:
            strategy_action = "Standard Neutral List Price Maintained"

        return {
            "category": category,
            "base_price": base_price,
            "recommended_dynamic_price": rounded_dynamic_price,
            "price_delta_dollars": round(rounded_dynamic_price - base_price, 2),
            "price_delta_pct": round(delta_vs_base_pct, 2),
            "multipliers": {
                "scarcity_multiplier": round(scarcity_multiplier, 3),
                "demand_surge_multiplier": round(surge_multiplier, 3),
            },
            "stock_context": {
                "current_stock": current_stock,
                "safety_stock": safety_stock,
                "stock_status": stock_status,
            },
            "economics": {
                "unit_cost": round(unit_cost, 2),
                "projected_gross_margin_dollars": round(rounded_dynamic_price - unit_cost, 2),
                "projected_gross_margin_pct": round(expected_margin_pct, 2),
                "expected_conversion_probability": round(conv_prob, 4),
            },
            "guardrails": {
                "price_floor": round(price_floor, 2),
                "price_ceiling": round(price_ceiling, 2),
                "competitor_constrained": competitor_constrained,
            },
            "pricing_action": strategy_action,
        }
