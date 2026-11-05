"""
Retail Cash Flow, Working Capital Cycle & Monte Carlo Financial Risk Stress Testing Engine.

Provides:
1. Enterprise Cash Conversion Cycle (CCC) Modeling:
   - DIO (Days Inventory Outstanding) = (Average Inventory / Annual COGS) * 365
   - DSO (Days Sales Outstanding) = (Accounts Receivable / Annual Gross Revenue) * 365
   - DPO (Days Payables Outstanding) = (Accounts Payable / Annual COGS) * 365
   - Net Cash Conversion Cycle (CCC) = DIO + DSO - DPO
2. Forward-Looking Monte Carlo Enterprise Liquidity Stress Testing (10,000 stochastic runs):
   - Stochastic demand contraction shocks, supplier COGS inflation, and OPEX volatility
   - Value at Risk (VaR 95% & 99%) and Conditional Value at Risk (CVaR / Expected Shortfall)
   - Solvency runway projection and probability of liquidity buffer depletion
3. Working Capital Optimization Simulator:
   - Simulates cash freed up from DIO acceleration, DPO renegotiation, and DSO compression
   - Assesses capital release and borrowing interest savings
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.analytics import RetailAnalyticsEngine


class FinancialRiskStressEngine:
    """
    Enterprise retail corporate treasury, working capital liquidity,
    and Monte Carlo financial solvency stress-testing engine.
    """

    # Baseline operating cost parameters across merchandise categories
    CATEGORY_COST_FACTORS = {
        "Beauty": {"gross_margin": 0.52, "avg_dio": 67.6, "credit_dso": 3.5, "supplier_dpo": 45.0},
        "Clothing": {"gross_margin": 0.46, "avg_dio": 76.0, "credit_dso": 3.0, "supplier_dpo": 42.0},
        "Electronics": {"gross_margin": 0.28, "avg_dio": 58.9, "credit_dso": 4.0, "supplier_dpo": 38.0},
    }

    # Baseline fixed annual retail operating overhead (rent, utilities, core payroll)
    BASE_ANNUAL_OVERHEAD = 120000.0

    def __init__(self, data_path: Optional[Path] = None, seed: int = 42):
        self.analytics = RetailAnalyticsEngine(data_path=data_path)
        self.df = self.analytics.df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.rng = np.random.default_rng(seed)

    def calculate_cash_conversion_cycle(self) -> Dict[str, Any]:
        """
        Computes storewide and category-level Cash Conversion Cycle (CCC = DIO + DSO - DPO)
        and working capital tied up in the business operating cycle.
        """
        store_revenue = float(self.df["Total Amount"].sum())
        total_cogs = 0.0
        total_inventory_val = 0.0
        total_ar_val = 0.0
        total_ap_val = 0.0

        category_breakdown = []
        for cat in sorted(self.df["Product Category"].unique()):
            cat_df = self.df[self.df["Product Category"] == cat]
            rev = float(cat_df["Total Amount"].sum())
            factors = self.CATEGORY_COST_FACTORS.get(
                cat, {"gross_margin": 0.40, "avg_dio": 65.0, "credit_dso": 3.0, "supplier_dpo": 40.0}
            )
            gm = factors["gross_margin"]
            cogs = rev * (1.0 - gm)
            dio = factors["avg_dio"]
            dso = factors["credit_dso"]
            dpo = factors["supplier_dpo"]

            ccc = dio + dso - dpo
            inv_capital = (dio / 365.0) * cogs
            ar_capital = (dso / 365.0) * rev
            ap_capital = (dpo / 365.0) * cogs
            net_wc = inv_capital + ar_capital - ap_capital

            total_cogs += cogs
            total_inventory_val += inv_capital
            total_ar_val += ar_capital
            total_ap_val += ap_capital

            category_breakdown.append({
                "category": cat,
                "annual_revenue": round(rev, 2),
                "annual_cogs": round(cogs, 2),
                "gross_margin_pct": round(gm * 100.0, 2),
                "days_inventory_outstanding_dio": round(dio, 1),
                "days_sales_outstanding_dso": round(dso, 1),
                "days_payables_outstanding_dpo": round(dpo, 1),
                "cash_conversion_cycle_days": round(ccc, 1),
                "inventory_capital_tied_up": round(inv_capital, 2),
                "accounts_receivable_tied_up": round(ar_capital, 2),
                "accounts_payable_liability": round(ap_capital, 2),
                "net_working_capital_required": round(net_wc, 2),
            })

        store_blended_dio = (total_inventory_val / total_cogs) * 365.0 if total_cogs > 0 else 0.0
        store_blended_dso = (total_ar_val / store_revenue) * 365.0 if store_revenue > 0 else 0.0
        store_blended_dpo = (total_ap_val / total_cogs) * 365.0 if total_cogs > 0 else 0.0
        store_blended_ccc = store_blended_dio + store_blended_dso - store_blended_dpo
        store_net_working_capital = total_inventory_val + total_ar_val - total_ap_val

        return {
            "store_total_revenue": round(store_revenue, 2),
            "store_total_cogs": round(total_cogs, 2),
            "store_blended_dio": round(store_blended_dio, 1),
            "store_blended_dso": round(store_blended_dso, 1),
            "store_blended_dpo": round(store_blended_dpo, 1),
            "store_blended_ccc_days": round(store_blended_ccc, 1),
            "total_inventory_investment": round(total_inventory_val, 2),
            "total_accounts_receivable": round(total_ar_val, 2),
            "total_accounts_payable": round(total_ap_val, 2),
            "net_working_capital_tied_up": round(store_net_working_capital, 2),
            "category_metrics": category_breakdown,
        }

    def run_monte_carlo_stress_test(
        self,
        n_simulations: int = 5000,
        demand_shock_mean_pct: float = -0.10,
        demand_shock_std_pct: float = 0.08,
        cogs_inflation_mean_pct: float = 0.06,
        cogs_inflation_std_pct: float = 0.04,
        dpo_contraction_days: float = 5.0,
        initial_cash_buffer: float = 60000.0,
    ) -> Dict[str, Any]:
        """
        Executes stochastic Monte Carlo enterprise liquidity and solvency stress testing.

        Simulates multi-period operating cash flows under correlated macroeconomic shocks:
        - Consumer demand contraction
        - Supplier COGS price inflation
        - Credit tightening / DPO contraction

        Parameters:
            n_simulations: Number of Monte Carlo trial paths (1,000 to 20,000)
            demand_shock_mean_pct: Mean consumer demand contraction (-0.10 = -10%)
            demand_shock_std_pct: Standard deviation of demand contraction
            cogs_inflation_mean_pct: Mean supplier inflation (+0.06 = +6%)
            cogs_inflation_std_pct: Standard deviation of inflation
            dpo_contraction_days: Supplier payment acceleration (days DPO reduced)
            initial_cash_buffer: Starting treasury cash reserves ($)
        """
        if n_simulations < 100 or n_simulations > 20000:
            raise ValueError("n_simulations must be between 100 and 20,000.")
        if initial_cash_buffer <= 0:
            raise ValueError("initial_cash_buffer must be positive.")

        base_ccc = self.calculate_cash_conversion_cycle()
        base_annual_rev = base_ccc["store_total_revenue"]
        base_annual_cogs = base_ccc["store_total_cogs"]
        base_annual_overhead = self.BASE_ANNUAL_OVERHEAD

        # Baseline baseline operating cash flow before shocks
        base_annual_ocf = base_annual_rev - base_annual_cogs - base_annual_overhead

        # Draw stochastic shock vectors
        demand_shocks = self.rng.normal(
            loc=demand_shock_mean_pct,
            scale=demand_shock_std_pct,
            size=n_simulations,
        )
        cogs_shocks = self.rng.normal(
            loc=cogs_inflation_mean_pct,
            scale=cogs_inflation_std_pct,
            size=n_simulations,
        )

        simulated_revenues = base_annual_rev * (1.0 + demand_shocks)
        simulated_cogs = base_annual_cogs * (1.0 + cogs_shocks) * (1.0 + demand_shocks)

        # Working capital drain from DPO contraction
        working_capital_drain = (dpo_contraction_days / 365.0) * simulated_cogs

        # Stressed Annual Operating Cash Flow
        stressed_ocfs = simulated_revenues - simulated_cogs - base_annual_overhead - working_capital_drain

        # End of year cash buffer
        ending_cash_balances = initial_cash_buffer + stressed_ocfs

        # Liquidity Shortfall / Insolvency Risk
        insolvency_mask = ending_cash_balances < 0
        insolvency_count = int(np.sum(insolvency_mask))
        insolvency_probability = (insolvency_count / n_simulations) * 100.0

        # Cash Runway in Months: for insolvent paths, estimated month of depletion
        monthly_burn = np.maximum(1000.0, -stressed_ocfs / 12.0)
        depletion_months = np.where(
            insolvency_mask,
            np.clip(initial_cash_buffer / monthly_burn, 1.0, 11.9),
            12.0,
        )
        avg_runway_months = float(np.mean(depletion_months))

        # Risk Metrics: Value at Risk (VaR) & Conditional Value at Risk (CVaR)
        # VaR alpha: percentile of loss (negative cash flow)
        var_95_cf = float(np.percentile(stressed_ocfs, 5.0))
        var_99_cf = float(np.percentile(stressed_ocfs, 1.0))

        cvar_95_cf = float(np.mean(stressed_ocfs[stressed_ocfs <= var_95_cf]))
        cvar_99_cf = float(np.mean(stressed_ocfs[stressed_ocfs <= var_99_cf]))

        # Empirical distribution histogram for UI rendering
        hist_counts, bin_edges = np.histogram(stressed_ocfs, bins=30)
        distribution_bins = [
            {
                "bin_start": round(float(bin_edges[i]), 2),
                "bin_end": round(float(bin_edges[i + 1]), 2),
                "frequency": int(hist_counts[i]),
            }
            for i in range(len(hist_counts))
        ]

        # Recommended minimum capital buffer to achieve < 1% insolvency probability
        # Required buffer = - (1st percentile OCF) if negative else $20,000 contingency
        recommended_buffer = max(20000.0, -var_99_cf + 15000.0)

        return {
            "simulations_count": n_simulations,
            "macro_assumptions": {
                "demand_shock_mean_pct": round(demand_shock_mean_pct * 100.0, 2),
                "cogs_inflation_mean_pct": round(cogs_inflation_mean_pct * 100.0, 2),
                "dpo_contraction_days": round(dpo_contraction_days, 1),
                "initial_cash_buffer": round(initial_cash_buffer, 2),
            },
            "baseline_financials": {
                "base_annual_revenue": round(base_annual_rev, 2),
                "base_annual_cogs": round(base_annual_cogs, 2),
                "base_annual_overhead": round(base_annual_overhead, 2),
                "base_operating_cash_flow": round(base_annual_ocf, 2),
            },
            "stressed_outcomes": {
                "mean_stressed_ocf": round(float(np.mean(stressed_ocfs)), 2),
                "median_stressed_ocf": round(float(np.median(stressed_ocfs)), 2),
                "min_stressed_ocf": round(float(np.min(stressed_ocfs)), 2),
                "max_stressed_ocf": round(float(np.max(stressed_ocfs)), 2),
                "var_95_cash_flow": round(var_95_cf, 2),
                "var_99_cash_flow": round(var_99_cf, 2),
                "cvar_95_expected_shortfall": round(cvar_95_cf, 2),
                "cvar_99_expected_shortfall": round(cvar_99_cf, 2),
            },
            "liquidity_and_solvency": {
                "insolvent_paths_count": insolvency_count,
                "insolvency_risk_pct": round(insolvency_probability, 2),
                "average_solvency_runway_months": round(avg_runway_months, 1),
                "minimum_safe_treasury_buffer_required": round(recommended_buffer, 2),
                "capital_buffer_surplus_or_deficit": round(initial_cash_buffer - recommended_buffer, 2),
            },
            "cash_flow_distribution": distribution_bins,
        }

    def simulate_working_capital_optimization(
        self,
        dio_reduction_days: float = 12.0,
        dpo_extension_days: float = 10.0,
        dso_reduction_days: float = 1.0,
        short_term_borrowing_rate_pct: float = 7.5,
    ) -> Dict[str, Any]:
        """
        Simulates commercial liquidity liberation and debt service savings
        by accelerating DIO, negotiating extended DPO terms, and compressing DSO.

        Parameters:
            dio_reduction_days: Days of inventory eliminated through rationalization
            dpo_extension_days: Extra supplier credit days negotiated
            dso_reduction_days: Days DSO reduced through instant electronic settlement
            short_term_borrowing_rate_pct: Annual credit facility interest rate saved
        """
        if dio_reduction_days < 0 or dpo_extension_days < 0 or dso_reduction_days < 0:
            raise ValueError("Day adjustment parameters cannot be negative.")
        if short_term_borrowing_rate_pct < 0:
            raise ValueError("borrowing rate cannot be negative.")

        base_ccc = self.calculate_cash_conversion_cycle()
        annual_cogs = base_ccc["store_total_cogs"]
        annual_rev = base_ccc["store_total_revenue"]
        current_wc = base_ccc["net_working_capital_tied_up"]
        current_ccc_days = base_ccc["store_blended_ccc_days"]

        # Dollars unlocked
        cash_from_inventory = (dio_reduction_days / 365.0) * annual_cogs
        cash_from_suppliers = (dpo_extension_days / 365.0) * annual_cogs
        cash_from_receivables = (dso_reduction_days / 365.0) * annual_rev

        total_liquidity_unlocked = cash_from_inventory + cash_from_suppliers + cash_from_receivables
        optimized_wc = max(0.0, current_wc - total_liquidity_unlocked)
        optimized_ccc_days = current_ccc_days - (dio_reduction_days + dso_reduction_days + dpo_extension_days)

        # Annual interest savings on working capital line of credit
        annual_interest_savings = total_liquidity_unlocked * (short_term_borrowing_rate_pct / 100.0)

        return {
            "baseline": {
                "current_ccc_days": current_ccc_days,
                "current_working_capital_tied_up": current_wc,
            },
            "adjustments": {
                "dio_reduction_days": dio_reduction_days,
                "dpo_extension_days": dpo_extension_days,
                "dso_reduction_days": dso_reduction_days,
            },
            "optimized_outcomes": {
                "optimized_ccc_days": round(optimized_ccc_days, 1),
                "ccc_reduction_days": round(current_ccc_days - optimized_ccc_days, 1),
                "cash_freed_from_inventory": round(cash_from_inventory, 2),
                "cash_freed_from_supplier_terms": round(cash_from_suppliers, 2),
                "cash_freed_from_ar_acceleration": round(cash_from_receivables, 2),
                "total_permanent_liquidity_unlocked": round(total_liquidity_unlocked, 2),
                "optimized_working_capital": round(optimized_wc, 2),
                "annual_borrowing_interest_savings": round(annual_interest_savings, 2),
            },
        }
