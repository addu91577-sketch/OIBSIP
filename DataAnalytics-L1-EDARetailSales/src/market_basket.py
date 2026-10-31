"""
Market Basket Association Rules & Cross-Merchandising Affinity Mining Engine.

Provides:
1. Multi-item transaction & session co-occurrence mining across merchandise categories.
2. Formal association rule evaluation: Support, Confidence, Lift, Leverage, Conviction.
3. Directional cross-selling propensity metrics and Chi-square independence testing.
4. Strategic cross-merchandising bundle generator with margin protection guardrails.
5. Demographic cohort category affinity cross-tabulation (Age Cohorts x Gender).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.analytics import RetailAnalyticsEngine


class MarketBasketAffinityEngine:
    """
    Market basket analysis, association rule mining, and merchandising synergy engine
    designed to break single-item transaction barriers and expand basket depth.
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.analytics = RetailAnalyticsEngine(data_path=data_path)
        self.df = self.analytics.df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.categories = sorted(self.df["Product Category"].unique().tolist())
        self.total_transactions = len(self.df)

    def get_category_summary(self) -> Dict[str, Any]:
        """
        Summarizes baseline category volume, revenue, and penetration rate.
        """
        cat_stats = (
            self.df.groupby("Product Category")
            .agg(
                transactions=("Transaction ID", "count"),
                total_units=("Quantity", "sum"),
                total_revenue=("Total Amount", "sum"),
                avg_price=("Price per Unit", "mean"),
            )
            .reset_index()
        )

        res = {}
        for _, row in cat_stats.iterrows():
            cat = row["Product Category"]
            txn_count = int(row["transactions"])
            penetration = txn_count / self.total_transactions
            res[cat] = {
                "transaction_count": txn_count,
                "penetration_rate": round(penetration, 4),
                "total_units": int(row["total_units"]),
                "total_revenue": round(float(row["total_revenue"]), 2),
                "avg_price_per_unit": round(float(row["avg_price"]), 2),
            }
        return res

    def build_cooccurrence_and_affinity_matrix(self) -> Dict[str, Any]:
        """
        Computes empirical cross-category co-occurrence, association rule metrics,
        and statistical affinity indicators across daily shopping occasions.
        """
        # Sessionize by Date and Demographics (shopping occasion proxy)
        daily_cat = (
            self.df.groupby(["Date", "Gender", "Product Category"])["Transaction ID"]
            .count()
            .unstack(fill_value=0)
        )
        total_sessions = len(daily_cat)

        # Baseline single category probabilities (Support)
        support = {}
        for cat in self.categories:
            if cat in daily_cat.columns:
                support[cat] = (daily_cat[cat] > 0).mean()
            else:
                support[cat] = 0.0

        pairwise_metrics = []
        for i, cat_a in enumerate(self.categories):
            for j, cat_b in enumerate(self.categories):
                if cat_a == cat_b:
                    continue

                col_a = daily_cat[cat_a] if cat_a in daily_cat.columns else pd.Series(0, index=daily_cat.index)
                col_b = daily_cat[cat_b] if cat_b in daily_cat.columns else pd.Series(0, index=daily_cat.index)

                cooccur_count = int(((col_a > 0) & (col_b > 0)).sum())
                supp_ab = cooccur_count / total_sessions if total_sessions > 0 else 0.0
                supp_a = support.get(cat_a, 0.0)
                supp_b = support.get(cat_b, 0.0)

                conf_a_to_b = (supp_ab / supp_a) if supp_a > 0 else 0.0
                conf_b_to_a = (supp_ab / supp_b) if supp_b > 0 else 0.0
                lift = (supp_ab / (supp_a * supp_b)) if (supp_a * supp_b) > 0 else 1.0
                leverage = supp_ab - (supp_a * supp_b)

                denom_conv = 1.0 - conf_a_to_b
                conviction = ((1.0 - supp_b) / denom_conv) if denom_conv > 1e-6 else 10.0

                # Contingency table for 2x2 Chi-Square test of independence
                n11 = cooccur_count
                n10 = int(((col_a > 0) & (col_b == 0)).sum())
                n01 = int(((col_a == 0) & (col_b > 0)).sum())
                n00 = total_sessions - (n11 + n10 + n01)

                contingency = np.array([[n11, n10], [n01, n00]])
                if contingency.min() >= 0 and contingency.sum() > 0:
                    chi2, p_val, _, _ = stats.chi2_contingency(contingency, correction=True)
                else:
                    chi2, p_val = 0.0, 1.0

                affinity_label = "Positive Synergy" if lift > 1.05 else ("Negative Cannibalization" if lift < 0.95 else "Neutral Independent")

                pairwise_metrics.append(
                    {
                        "antecedent": cat_a,
                        "consequent": cat_b,
                        "cooccurrence_sessions": cooccur_count,
                        "support_antecedent": round(float(supp_a), 4),
                        "support_consequent": round(float(supp_b), 4),
                        "support_joint": round(float(supp_ab), 4),
                        "confidence": round(float(conf_a_to_b), 4),
                        "lift": round(float(lift), 3),
                        "leverage": round(float(leverage), 4),
                        "conviction": round(float(conviction), 3),
                        "chi2_statistic": round(float(chi2), 3),
                        "p_value": round(float(p_val), 4),
                        "is_statistically_significant": bool(p_val < 0.05),
                        "affinity_nature": affinity_label,
                    }
                )

        return {
            "total_shopping_sessions": total_sessions,
            "individual_support": {k: round(v, 4) for k, v in support.items()},
            "rules": pairwise_metrics,
        }

    def mine_association_rules(
        self,
        min_support: float = 0.05,
        min_confidence: float = 0.20,
        min_lift: float = 0.80,
    ) -> List[Dict[str, Any]]:
        """
        Filters and ranks association rules meeting specified minimum thresholds.
        """
        matrix = self.build_cooccurrence_and_affinity_matrix()
        rules = matrix["rules"]

        filtered = [
            r
            for r in rules
            if (
                r["support_joint"] >= min_support
                and r["confidence"] >= min_confidence
                and r["lift"] >= min_lift
            )
        ]

        # Rank by lift descending, then confidence descending
        filtered.sort(key=lambda x: (x["lift"], x["confidence"]), reverse=True)
        return filtered

    def recommend_merchandising_combos(
        self,
        target_category: str,
        discount_rate: float = 0.12,
        gross_margin_pct: float = 0.40,
    ) -> List[Dict[str, Any]]:
        """
        Generates optimal cross-department promotional bundle pairings for a target category.
        Computes bundled retail prices, margin protection, and estimated profit lift.
        """
        if target_category not in self.categories:
            raise ValueError(
                f"Unknown target category '{target_category}'. Available: {self.categories}"
            )

        if not (0.01 <= discount_rate <= 0.40):
            raise ValueError(f"Discount rate must be between 0.01 and 0.40, got {discount_rate}")

        summary = self.get_category_summary()
        matrix = self.build_cooccurrence_and_affinity_matrix()
        rules = [r for r in matrix["rules"] if r["antecedent"] == target_category]

        target_avg_price = summary[target_category]["avg_price_per_unit"]

        recommendations = []
        for r in rules:
            consequent = r["consequent"]
            partner_avg_price = summary[consequent]["avg_price_per_unit"]

            orig_bundle_price = target_avg_price + partner_avg_price
            discounted_bundle_price = orig_bundle_price * (1.0 - discount_rate)
            customer_savings = orig_bundle_price - discounted_bundle_price

            # Margin analysis
            cogs = orig_bundle_price * (1.0 - gross_margin_pct)
            bundle_gross_profit = discounted_bundle_price - cogs
            retained_margin_pct = (bundle_gross_profit / discounted_bundle_price) if discounted_bundle_price > 0 else 0.0

            # Commercial synergy score (Lift * Confidence * Margin Retained)
            synergy_score = r["lift"] * r["confidence"] * (retained_margin_pct * 100.0)

            recommendations.append(
                {
                    "base_category": target_category,
                    "complementary_category": consequent,
                    "association_lift": r["lift"],
                    "purchase_confidence": r["confidence"],
                    "affinity_nature": r["affinity_nature"],
                    "original_bundle_price": round(orig_bundle_price, 2),
                    "promotional_discount_pct": round(discount_rate * 100.0, 1),
                    "discounted_bundle_price": round(discounted_bundle_price, 2),
                    "customer_savings": round(customer_savings, 2),
                    "estimated_retained_margin_pct": round(retained_margin_pct * 100.0, 2),
                    "synergy_score": round(synergy_score, 2),
                    "recommended_action": f"Pair {target_category} with {consequent} at {discount_rate*100:.0f}% off bundle discount.",
                }
            )

        recommendations.sort(key=lambda x: x["synergy_score"], reverse=True)
        return recommendations

    def demographic_affinity_breakdown(self) -> Dict[str, Any]:
        """
        Cross-analyzes category preferences and joint affinities across demographic age cohorts.
        """
        # Create standard age cohorts
        df = self.df.copy()
        bins = [17, 25, 35, 50, 65]
        labels = ["Young Adults (18-25)", "Early Career (26-35)", "Mid Career (36-50)", "Mature / Seniors (51-64)"]
        df["Age_Cohort"] = pd.cut(df["Age"], bins=bins, labels=labels)

        cohort_data = {}
        for cohort in labels:
            c_df = df[df["Age_Cohort"] == cohort]
            if len(c_df) == 0:
                continue

            cat_counts = c_df["Product Category"].value_counts().to_dict()
            total_orders = len(c_df)
            top_cat = c_df["Product Category"].mode()[0] if not c_df.empty else "None"
            avg_spend = float(c_df["Total Amount"].mean())

            # Category share breakdown
            shares = {k: round(v / total_orders, 4) for k, v in cat_counts.items()}

            cohort_data[cohort] = {
                "total_transactions": total_orders,
                "dominant_category": top_cat,
                "average_basket_spend": round(avg_spend, 2),
                "category_shares": shares,
                "multi_category_affinity_index": round(
                    1.0 - (max(shares.values()) if shares else 1.0), 3
                ),
            }

        return {
            "cohort_count": len(cohort_data),
            "cohort_breakdown": cohort_data,
        }
