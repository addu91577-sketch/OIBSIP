"""
Strategic Business Recommendations & Cross-Department Bundle Engine.

Operationalizes insights from EDA Day 6 into algorithmic recommendations,
cross-category bundle curation, and financial impact calculators.
"""

from typing import Dict, Any, List


class BusinessRecommendationEngine:
    """Delivers data-driven strategic retail recommendations and dynamic bundling logic."""

    @staticmethod
    def get_strategic_recommendations() -> List[Dict[str, Any]]:
        """Returns the 5 core strategic business initiatives identified in EDA."""
        return [
            {
                "id": "REC-01",
                "title": "VIP Concierge & Whale Retention Engine",
                "target_segment": "VIP Customers (Spend >= $1,000)",
                "empirical_driver": (
                    "Top 20.2% of transactions generate 62.89% of enterprise revenue ($286,800.00) "
                    "with an average order value of $1,419.80."
                ),
                "actionable_initiatives": [
                    "Institutionalize invite-only 'Black Tier' VIP Loyalty Program.",
                    "Provide dedicated clienteling, white-glove support, and complimentary priority shipping.",
                    "Automate 90-day re-engagement triggers with tailored luxury previews.",
                ],
                "projected_financial_impact": (
                    "A 5% reduction in VIP customer churn preserves ~$14,300 in annual recurring gross margin."
                ),
                "kpi_target": "VIP Retention Rate >= 85%",
            },
            {
                "id": "REC-02",
                "title": "Omni-Channel Day-of-Week Media & Staffing Synchronization",
                "target_segment": "Beauty & Tech Shoppers Across Weekly Cycle",
                "empirical_driver": (
                    "Beauty AOV peaks on Mondays ($623.59) and Fridays ($577.16). "
                    "Electronics surges +23.0% on weekends to peak on Saturday ($563.68) alongside Clothing ($533.64)."
                ),
                "actionable_initiatives": [
                    "Concentrate paid ads for Beauty on Sunday night/Monday morning and Thursday night/Friday morning.",
                    "Shift paid ad spend Friday afternoon & Saturday toward high-ticket consumer electronics.",
                    "Align floor sales staff to Saturdays ($78,815 revenue); schedule warehouse/admin on Thursdays ($53,835).",
                ],
                "projected_financial_impact": (
                    "Lowers digital Customer Acquisition Cost (CAC) by 12–18% through intent-aligned ad timing."
                ),
                "kpi_target": "Weekend Conversion Lift >= 15%",
            },
            {
                "id": "REC-03",
                "title": "Cross-Department Bundles to Breach the 2.5-Unit Inelasticity Cap",
                "target_segment": "All Catalog Buyers",
                "empirical_driver": (
                    "Basket units are completely inelastic across price points (r = +0.017, p = 0.580), "
                    "hovering near 2.5 units whether items cost $25 or $500."
                ),
                "actionable_initiatives": [
                    "Curate cross-department sets (e.g., 'Executive Tech & Style' or 'Total Self-Care').",
                    "Deploy tiered bundle pricing: 'Buy 3 items, save 10%; buy 4 items, save 15%'.",
                    "Add automated cross-category recommendation cards at checkout.",
                ],
                "projected_financial_impact": (
                    "Increasing average basket units from 2.51 to 3.00 delivers +$89,000.00 (+19.5%) "
                    "in incremental annual revenue with zero new customer acquisition cost."
                ),
                "kpi_target": "Storewide AOV >= $520.00; Units/Basket >= 3.0",
            },
            {
                "id": "REC-04",
                "title": "Capitalize on High-Ticket Personas (Young Adult VIPs & Male Beauty Gifting)",
                "target_segment": "18–25 Cohort & Male Cosmetics Gifting",
                "empirical_driver": (
                    "18–25 cohort delivers highest VIP conversion rate (25.44%). "
                    "Male shoppers generate $637.32 Beauty AOV in 18–25 (+51.8% over females). "
                    "Older adults (51–64) buy the highest volume of VIP electronics (26 orders)."
                ),
                "actionable_initiatives": [
                    "Design curated 'Luxury Gifts for Her' landing pages targeted at male shoppers ahead of holidays.",
                    "Integrate Buy Now, Pay Later (BNPL) options (Klarna/Affirm) to eliminate liquidity friction for young adults.",
                    "Promote premium build quality and technical longevity in Electronics campaigns targeted at mature adults.",
                ],
                "projected_financial_impact": (
                    "Expands young adult checkout completion by 10–15% and captures high-margin luxury gift conversions."
                ),
                "kpi_target": "Cart Abandonment Drop <= 22%",
            },
            {
                "id": "REC-05",
                "title": "Pre-Q4 Working Capital Buffers & Late-Summer Promotional Hedging",
                "target_segment": "Procurement, Supply Chain & Inventory Operations",
                "empirical_driver": (
                    "Sales drop sharply in September ($23,620) before surging +97.2% in October ($46,580) "
                    "into a $126,190 Q4 finish. Beauty acts as an organic counter-cyclical hedge in July ($16,090)."
                ),
                "actionable_initiatives": [
                    "Execute mid-tier Beauty and summer promotional sales in August/September to stabilize cash flow.",
                    "Submit vendor purchase orders for high-margin $300 and $500 items by late August.",
                    "Establish credit facilities to prevent inventory stockouts ahead of holiday volume surges.",
                ],
                "projected_financial_impact": (
                    "Protects against holiday inventory stockouts, safeguarding ~$20,000–$35,000 in high-margin Q4 revenue."
                ),
                "kpi_target": "Zero Out-of-Stock Incidents during Q4 Holiday Surge",
            },
        ]

    @staticmethod
    def generate_bundle(
        primary_category: str,
        target_units: int = 3,
    ) -> Dict[str, Any]:
        """
        Dynamically designs an optimized cross-department bundle based on customer preference.
        Applies 10% discount for 3 units, 15% discount for 4 units.
        """
        category_complement_map = {
            "Electronics": [
                {"category": "Clothing", "item": "Executive Casual Apparel", "unit_price": 50.0},
                {"category": "Beauty", "item": "Revitalizing Skincare Kit", "unit_price": 50.0},
                {"category": "Electronics", "item": "High-Fidelity Audio Accessory", "unit_price": 300.0},
                {"category": "Clothing", "item": "Designer Travel Overcoat", "unit_price": 300.0},
            ],
            "Clothing": [
                {"category": "Electronics", "item": "Smart Wearable Fitness Tracker", "unit_price": 300.0},
                {"category": "Beauty", "item": "Luxury Eau De Parfum", "unit_price": 50.0},
                {"category": "Clothing", "item": "Tailored Formalwear Ensemble", "unit_price": 300.0},
                {"category": "Electronics", "item": "Noise-Cancelling Wireless Earbuds", "unit_price": 50.0},
            ],
            "Beauty": [
                {"category": "Clothing", "item": "Silk Loungewear Collection", "unit_price": 50.0},
                {"category": "Electronics", "item": "Sonic Cleansing & Wellness Device", "unit_price": 300.0},
                {"category": "Beauty", "item": "Botanical Hydration Day Cream", "unit_price": 50.0},
                {"category": "Clothing", "item": "Merino Wool Cashmere Wrap", "unit_price": 300.0},
            ],
        }

        items = category_complement_map.get(
            primary_category, category_complement_map["Electronics"]
        )[: max(1, min(target_units, 4))]

        raw_total = sum(item["unit_price"] for item in items)
        discount_pct = 15.0 if target_units >= 4 else (10.0 if target_units >= 3 else 0.0)
        discount_multiplier = 1.0 - (discount_pct / 100.0)
        bundle_price = round(raw_total * discount_multiplier, 2)
        customer_savings = round(raw_total - bundle_price, 2)

        return {
            "bundle_name": f"{primary_category} Cross-Department Synergy Set",
            "primary_category": primary_category,
            "items_count": len(items),
            "bundle_items": items,
            "original_total_price": raw_total,
            "discount_percentage": discount_pct,
            "discounted_bundle_price": bundle_price,
            "customer_savings": customer_savings,
            "business_upsell_value": (
                f"Elevates unit count to {len(items)} items (+{len(items) - 2.51:.2f} above baseline average)."
            ),
        }
