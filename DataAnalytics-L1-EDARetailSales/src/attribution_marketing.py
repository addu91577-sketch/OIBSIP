"""
Omnichannel Multi-Touch Attribution & Marketing Channel ROI Optimization Engine.

Provides:
1. Rule-based heuristic attribution models:
   - First-Touch: 100% conversion credit to top-of-funnel acquisition channel.
   - Last-Touch: 100% conversion credit to final converting channel.
   - Linear: Equal credit allocation across all customer journey touchpoints.
   - Time-Decay: Exponential half-life decay prioritizing recent touchpoints.
   - Position-Based (U-Shaped): 40% First, 40% Last, 20% divided among middle touchpoints.
2. Algorithmic Game-Theoretic Shapley Value Attribution:
   - Evaluates cooperative coalition marginal contributions across channels.
3. First-Order Markov Transition Chain Attribution:
   - Computes state transition matrix, absorbing conversion states, and channel removal effects.
4. Channel Economics & Efficiency Metrics:
   - Attributed revenue, attributed conversions, Cost-Per-Acquisition (CPA), Return on Ad Spend (ROAS).
5. Markowitz Modern Portfolio Theory (MPT) Optimal Budget Allocation:
   - Quadratic optimization to maximize storewide ROAS and minimize risk variance.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.analytics import RetailAnalyticsEngine


class AttributionMarketingEngine:
    """
    Omnichannel Multi-Touch Marketing Attribution & Budget Allocation Engine.
    Enables retail marketing executives to quantify channel contribution and optimize spend.
    """

    CHANNELS = [
        "Organic Search (SEO)",
        "Paid Search (Google Ads)",
        "Social Media (Meta/TikTok)",
        "Email / CRM Retargeting",
        "Direct / In-Store",
    ]

    # Benchmark marketing spends across channels for 1,000 transactions
    DEFAULT_CHANNEL_SPENDS = {
        "Organic Search (SEO)": 8000.0,
        "Paid Search (Google Ads)": 28000.0,
        "Social Media (Meta/TikTok)": 32000.0,
        "Email / CRM Retargeting": 7000.0,
        "Direct / In-Store": 5000.0,
    }

    def __init__(
        self,
        data_path: Optional[Path] = None,
        half_life_days: float = 7.0,
        channel_spends: Optional[Dict[str, float]] = None,
    ):
        self.analytics = RetailAnalyticsEngine(data_path=data_path)
        self.df = self.analytics.df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.half_life_days = half_life_days
        self.channel_spends = channel_spends or self.DEFAULT_CHANNEL_SPENDS.copy()

        # Build synthetic deterministic customer journey touchpoint history
        self._journeys_df: Optional[pd.DataFrame] = None
        self._build_customer_journeys()

    def _build_customer_journeys(self) -> None:
        """
        Synthesizes deterministic multi-touch customer journey paths for all 1,000 transactions
        grounded in customer age, product category, and purchase date.
        """
        np.random.seed(42)
        records = []

        for idx, row in self.df.iterrows():
            cust_id = row["Customer ID"]
            tx_id = int(row["Transaction ID"])
            tx_date = row["Date"]
            category = row["Product Category"]
            age = int(row["Age"])
            total_amt = float(row["Total Amount"])

            # Determine path length (1 to 4 touchpoints) based on basket size
            if total_amt > 1000:
                n_touchpoints = 4
            elif total_amt > 400:
                n_touchpoints = 3
            elif total_amt > 100:
                n_touchpoints = 2
            else:
                n_touchpoints = 1

            # Channel probabilities conditioned on demographics and category
            if age < 35:
                probs = [0.15, 0.25, 0.40, 0.10, 0.10]
            elif age < 50:
                probs = [0.25, 0.30, 0.20, 0.15, 0.10]
            else:
                probs = [0.20, 0.20, 0.10, 0.30, 0.20]

            if category == "Electronics":
                probs[1] += 0.10  # Higher paid search for electronics
                probs[2] -= 0.10
            elif category == "Clothing":
                probs[2] += 0.10  # Higher social media for apparel
                probs[1] -= 0.10

            # Normalize probabilities
            probs = np.array(probs)
            probs = np.maximum(probs, 0.05)
            probs = probs / probs.sum()

            # Deterministic selection based on transaction hash
            tx_seed = (tx_id * 10007 + age * 31) % 2**32
            rng = np.random.RandomState(tx_seed)

            # Sample touchpoints with chronological offsets (days before conversion)
            chosen_channels = rng.choice(self.CHANNELS, size=n_touchpoints, replace=True, p=probs).tolist()
            
            # Days offset before conversion (e.g., 0 days for converting touchpoint)
            days_before = sorted([rng.randint(0, 18) for _ in range(n_touchpoints - 1)], reverse=True) + [0]

            records.append({
                "transaction_id": tx_id,
                "customer_id": cust_id,
                "date": tx_date,
                "category": category,
                "total_amount": total_amt,
                "path_length": n_touchpoints,
                "channels": chosen_channels,
                "days_before": days_before,
                "converted": True,
            })

        self._journeys_df = pd.DataFrame(records)

    def get_heuristic_attribution_summary(self) -> Dict[str, Any]:
        """
        Computes credit allocation across 5 standard heuristic attribution models:
        First-Touch, Last-Touch, Linear, Time-Decay, and Position-Based (U-Shaped).
        """
        results = {
            "First-Touch": {c: 0.0 for c in self.CHANNELS},
            "Last-Touch": {c: 0.0 for c in self.CHANNELS},
            "Linear": {c: 0.0 for c in self.CHANNELS},
            "Time-Decay": {c: 0.0 for c in self.CHANNELS},
            "Position-Based": {c: 0.0 for c in self.CHANNELS},
        }
        rev_results = {
            "First-Touch": {c: 0.0 for c in self.CHANNELS},
            "Last-Touch": {c: 0.0 for c in self.CHANNELS},
            "Linear": {c: 0.0 for c in self.CHANNELS},
            "Time-Decay": {c: 0.0 for c in self.CHANNELS},
            "Position-Based": {c: 0.0 for c in self.CHANNELS},
        }

        decay_const = math.log(2) / max(self.half_life_days, 1.0)

        for _, row in self._journeys_df.iterrows():
            channels = row["channels"]
            days = row["days_before"]
            amt = row["total_amount"]
            k = len(channels)

            # 1. First-Touch
            c_first = channels[0]
            results["First-Touch"][c_first] += 1.0
            rev_results["First-Touch"][c_first] += amt

            # 2. Last-Touch
            c_last = channels[-1]
            results["Last-Touch"][c_last] += 1.0
            rev_results["Last-Touch"][c_last] += amt

            # 3. Linear
            weight_linear = 1.0 / k
            for c in channels:
                results["Linear"][c] += weight_linear
                rev_results["Linear"][c] += amt * weight_linear

            # 4. Time-Decay
            weights_td = [math.exp(-decay_const * d) for d in days]
            sum_td = sum(weights_td)
            if sum_td > 0:
                norm_td = [w / sum_td for w in weights_td]
            else:
                norm_td = [1.0 / k] * k
            for c, w in zip(channels, norm_td):
                results["Time-Decay"][c] += w
                rev_results["Time-Decay"][c] += amt * w

            # 5. Position-Based (U-Shaped: 40/40/20)
            if k == 1:
                weights_pb = [1.0]
            elif k == 2:
                weights_pb = [0.5, 0.5]
            else:
                mid_weight = 0.20 / (k - 2)
                weights_pb = [0.40] + [mid_weight] * (k - 2) + [0.40]

            for c, w in zip(channels, weights_pb):
                results["Position-Based"][c] += w
                rev_results["Position-Based"][c] += amt * w

        # Round and build structured output
        models_summary = {}
        for model in results.keys():
            channel_data = []
            for c in self.CHANNELS:
                orders = round(results[model][c], 2)
                revenue = round(rev_results[model][c], 2)
                spend = self.channel_spends.get(c, 1000.0)
                roas = round(revenue / spend, 2) if spend > 0 else 0.0
                cpa = round(spend / orders, 2) if orders > 0 else 0.0
                channel_data.append({
                    "channel": c,
                    "attributed_orders": orders,
                    "attributed_revenue": revenue,
                    "spend": spend,
                    "roas": roas,
                    "cpa": cpa,
                    "share_of_orders_pct": round(orders / len(self._journeys_df) * 100, 2),
                })
            models_summary[model] = channel_data

        return {
            "total_transactions": len(self._journeys_df),
            "total_revenue": round(float(self._journeys_df["total_amount"].sum()), 2),
            "half_life_days": self.half_life_days,
            "models": models_summary,
        }

    def compute_markov_attribution(self) -> Dict[str, Any]:
        """
        Computes algorithmic First-Order Markov Chain attribution with Removal Effects.
        Models transitions between touchpoint states, Start, and Conversion.
        Removal effect: R(c) = (P(conv) - P(conv | remove c)) / P(conv).
        """
        # Step 1: Count transitions between states
        states = ["(start)"] + self.CHANNELS + ["(conversion)", "(null)"]
        trans_counts = {s1: {s2: 0 for s2 in states} for s1 in states}

        np.random.seed(42)
        for _, row in self._journeys_df.iterrows():
            path = ["(start)"] + row["channels"] + ["(conversion)"]
            for i in range(len(path) - 1):
                trans_counts[path[i]][path[i + 1]] += 1

        # Add simulated non-converting (null) customer journeys (drop-offs)
        # 400 drop-off sessions to create realistic absorption probabilities
        for i in range(400):
            rng = np.random.RandomState(i + 999)
            k = rng.randint(1, 3)
            sampled = rng.choice(self.CHANNELS, size=k).tolist()
            path = ["(start)"] + sampled + ["(null)"]
            for j in range(len(path) - 1):
                trans_counts[path[j]][path[j + 1]] += 1

        # Step 2: Compute transition probability matrix
        trans_matrix = {}
        for s1 in states:
            total = sum(trans_counts[s1].values())
            if total > 0:
                trans_matrix[s1] = {s2: trans_counts[s1][s2] / total for s2 in states}
            else:
                # Absorbing state
                trans_matrix[s1] = {s2: (1.0 if s1 == s2 else 0.0) for s2 in states}

        def simulate_conversion_prob(matrix: Dict[str, Dict[str, float]], n_sims: int = 10000) -> float:
            """Estimates probability of reaching (conversion) from (start) via random walks."""
            conversions = 0
            rng = np.random.RandomState(123)
            for _ in range(n_sims):
                curr = "(start)"
                steps = 0
                while curr not in ["(conversion)", "(null)"] and steps < 20:
                    probs = [matrix[curr][s] for s in states]
                    s_sum = sum(probs)
                    if s_sum == 0:
                        break
                    norm_probs = [p / s_sum for p in probs]
                    curr = rng.choice(states, p=norm_probs)
                    steps += 1
                if curr == "(conversion)":
                    conversions += 1
            return conversions / n_sims

        base_p_conv = simulate_conversion_prob(trans_matrix)

        # Step 3: Compute Removal Effect for each channel
        removal_effects = {}
        for c in self.CHANNELS:
            # Create modified transition matrix with channel c removed (redirected to (null))
            mod_matrix = {s1: dict(trans_matrix[s1]) for s1 in states}
            for s1 in states:
                prob_to_c = mod_matrix[s1].get(c, 0.0)
                if prob_to_c > 0:
                    mod_matrix[s1][c] = 0.0
                    mod_matrix[s1]["(null)"] += prob_to_c

            p_conv_removed = simulate_conversion_prob(mod_matrix)
            r_c = max(0.0, (base_p_conv - p_conv_removed) / max(base_p_conv, 1e-6))
            removal_effects[c] = r_c

        # Step 4: Normalize removal effects to attribute total orders and revenue
        total_removal = sum(removal_effects.values())
        norm_shares = {c: (removal_effects[c] / total_removal if total_removal > 0 else 1.0 / len(self.CHANNELS)) for c in self.CHANNELS}

        total_rev = float(self._journeys_df["total_amount"].sum())
        total_orders = len(self._journeys_df)

        markov_results = []
        for c in self.CHANNELS:
            share = norm_shares[c]
            attr_orders = round(total_orders * share, 2)
            attr_rev = round(total_rev * share, 2)
            spend = self.channel_spends.get(c, 1000.0)
            roas = round(attr_rev / spend, 2) if spend > 0 else 0.0
            cpa = round(spend / attr_orders, 2) if attr_orders > 0 else 0.0

            markov_results.append({
                "channel": c,
                "removal_effect": round(removal_effects[c], 4),
                "attributed_share_pct": round(share * 100, 2),
                "attributed_orders": attr_orders,
                "attributed_revenue": attr_rev,
                "spend": spend,
                "roas": roas,
                "cpa": cpa,
            })

        return {
            "baseline_conversion_probability": round(base_p_conv, 4),
            "channels_evaluated": len(self.CHANNELS),
            "channel_attribution": markov_results,
        }

    def compute_shapley_attribution(self) -> Dict[str, Any]:
        """
        Computes cooperative Game-Theoretic Shapley Value Attribution.
        Evaluates marginal contribution of each channel across all 2^N coalition subsets.
        """
        # Build binary feature vectors of channel presence for all journeys
        n_channels = len(self.CHANNELS)
        total_rev = float(self._journeys_df["total_amount"].sum())

        # Coalition values v(S) represented as conversion revenue driven by subset of channels
        coalition_rev = {}
        # Precompute subsets
        for i in range(1 << n_channels):
            active_channels = set([self.CHANNELS[j] for j in range(n_channels) if (i & (1 << j))])
            coalition_rev[frozenset(active_channels)] = 0.0

        for _, row in self._journeys_df.iterrows():
            path_channels = frozenset(row["channels"])
            amt = row["total_amount"]
            # Any coalition that contains all path_channels captures this conversion
            for coalition in coalition_rev.keys():
                if path_channels.issubset(coalition):
                    coalition_rev[coalition] += amt

        # Compute Shapley Value phi_i for each channel
        shapley_values = {c: 0.0 for c in self.CHANNELS}
        channel_set = set(self.CHANNELS)

        for c in self.CHANNELS:
            other_channels = channel_set - {c}
            others_list = list(other_channels)
            m = len(others_list)
            
            # Iterate through all coalitions S of other channels
            for i in range(1 << m):
                S = frozenset([others_list[j] for j in range(m) if (i & (1 << j))])
                S_with_c = S | {c}
                marginal_contribution = coalition_rev[S_with_c] - coalition_rev[S]
                
                # Weight: |S|! * (n - |S| - 1)! / n!
                s_size = len(S)
                weight = (math.factorial(s_size) * math.factorial(n_channels - s_size - 1)) / math.factorial(n_channels)
                shapley_values[c] += weight * marginal_contribution

        # Normalize Shapley values to match total revenue
        sum_shapley = sum(shapley_values.values())
        normalized_shapley = {}
        for c in self.CHANNELS:
            normalized_shapley[c] = (shapley_values[c] / sum_shapley * total_rev) if sum_shapley > 0 else (total_rev / n_channels)

        shapley_results = []
        for c in self.CHANNELS:
            rev = round(normalized_shapley[c], 2)
            share_pct = round(rev / total_rev * 100, 2) if total_rev > 0 else 0.0
            spend = self.channel_spends.get(c, 1000.0)
            roas = round(rev / spend, 2) if spend > 0 else 0.0

            shapley_results.append({
                "channel": c,
                "raw_shapley_value": round(shapley_values[c], 2),
                "attributed_revenue": rev,
                "share_pct": share_pct,
                "spend": spend,
                "roas": roas,
            })

        return {
            "total_attributed_revenue": round(total_rev, 2),
            "shapley_attribution": shapley_results,
        }

    def optimize_marketing_budget(
        self,
        total_budget: Optional[float] = None,
        risk_aversion: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Executes Modern Portfolio Theory (Markowitz MPT) quadratic budget optimization.
        Maximizes expected storewide ROAS while penalizing variance/risk across channel returns.

        Parameters:
        - total_budget: Total marketing budget to allocate (defaults to sum of current spends).
        - risk_aversion: Penalty coefficient gamma on return volatility (0.0 = pure return, 1.0 = risk-averse).
        """
        current_spends = [self.channel_spends.get(c, 10000.0) for c in self.CHANNELS]
        tot_budget = total_budget if total_budget is not None and total_budget > 0 else sum(current_spends)

        # Baseline expected ROAS per channel from Markov attribution
        markov_data = self.compute_markov_attribution()["channel_attribution"]
        expected_roas = np.array([m["roas"] for m in markov_data])

        # Covariance matrix of channel returns (empirical retail volatility structure)
        # Channels have low-to-moderate cross-correlation
        volatilities = np.array([0.15, 0.25, 0.35, 0.12, 0.08])
        corr_matrix = np.array([
            [1.00, 0.40, 0.25, 0.15, 0.10],
            [0.40, 1.00, 0.50, 0.30, 0.15],
            [0.25, 0.50, 1.00, 0.20, 0.10],
            [0.15, 0.30, 0.20, 1.00, 0.30],
            [0.10, 0.15, 0.10, 0.30, 1.00],
        ])
        cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

        # Objective function: Minimize -(w^T * mu - (gamma / 2) * w^T * Sigma * w)
        n = len(self.CHANNELS)
        def objective(weights: np.ndarray) -> float:
            port_return = np.dot(weights, expected_roas)
            port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            return -(port_return - 0.5 * risk_aversion * port_variance)

        # Constraints: sum(w) = 1, each w in [0.05, 0.45] (diversification guardrails)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.05, 0.45) for _ in range(n)]
        init_weights = np.ones(n) / n

        opt_res = minimize(objective, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)

        opt_weights = opt_res.x if opt_res.success else init_weights
        opt_spends = opt_weights * tot_budget

        # Projected performance under optimal allocation
        baseline_tot_spend = sum(current_spends)
        baseline_proj_rev = sum([m["attributed_revenue"] for m in markov_data])
        baseline_blended_roas = baseline_proj_rev / baseline_tot_spend if baseline_tot_spend > 0 else 0.0

        projected_rev_opt = np.sum(opt_spends * expected_roas)
        projected_blended_roas = projected_rev_opt / tot_budget if tot_budget > 0 else 0.0
        revenue_uplift = projected_rev_opt - baseline_proj_rev
        uplift_pct = (revenue_uplift / baseline_proj_rev * 100) if baseline_proj_rev > 0 else 0.0

        allocation_comparison = []
        for i, c in enumerate(self.CHANNELS):
            curr_s = current_spends[i]
            opt_s = round(float(opt_spends[i]), 2)
            curr_share = round((curr_s / baseline_tot_spend) * 100, 2) if baseline_tot_spend > 0 else 0.0
            opt_share = round(float(opt_weights[i]) * 100, 2)
            delta_spend = round(opt_s - curr_s, 2)

            allocation_comparison.append({
                "channel": c,
                "current_spend": curr_s,
                "current_share_pct": curr_share,
                "optimal_spend": opt_s,
                "optimal_share_pct": opt_share,
                "spend_delta": delta_spend,
                "expected_roas": round(float(expected_roas[i]), 2),
                "action": "Increase Spend" if delta_spend > 0 else ("Decrease Spend" if delta_spend < 0 else "Maintain"),
            })

        return {
            "optimization_status": "CONVERGED" if opt_res.success else "FALLBACK",
            "risk_aversion_parameter": risk_aversion,
            "total_budget": round(tot_budget, 2),
            "baseline_revenue": round(baseline_proj_rev, 2),
            "projected_optimal_revenue": round(float(projected_rev_opt), 2),
            "projected_revenue_uplift": round(float(revenue_uplift), 2),
            "uplift_pct": round(float(uplift_pct), 2),
            "baseline_blended_roas": round(baseline_blended_roas, 2),
            "projected_blended_roas": round(float(projected_blended_roas), 2),
            "channel_allocations": allocation_comparison,
        }
