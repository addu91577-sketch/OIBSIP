"""
Bayesian & Frequentist A/B Testing & Sequential Experimentation Engine.

Provides:
1. Frequentist Welch's t-test and non-parametric Mann-Whitney U testing for continuous retail spend
2. Two-proportion z-test and Chi-square test for conversion and adoption rates
3. Bayesian Beta-Binomial conjugate posterior simulation with Probability of Being Best and Expected Loss
4. Bayesian continuous revenue testing with Monte Carlo posterior lift distributions
5. Statistical power analysis and Minimum Detectable Effect (MDE) sample size matrix
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import stats


class RetailABTestingEngine:
    """
    Dual Frequentist-Bayesian statistical experimentation suite
    for evaluating retail pricing, promotional campaigns, and bundle interventions.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def evaluate_continuous_ab(
        self,
        control_data: List[float],
        treatment_data: List[float],
        metric_name: str = "Basket Spend",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Executes dual parametric (Welch's t-test) and non-parametric (Mann-Whitney U)
        tests on continuous retail metrics (e.g. basket spend or revenue).
        """
        c_arr = np.asarray(control_data, dtype=float)
        t_arr = np.asarray(treatment_data, dtype=float)

        if len(c_arr) < 2 or len(t_arr) < 2:
            raise ValueError("Both control and treatment groups must contain at least 2 observations.")

        n_c, n_t = len(c_arr), len(t_arr)
        mean_c, mean_t = float(np.mean(c_arr)), float(np.mean(t_arr))
        std_c, std_t = float(np.std(c_arr, ddof=1)), float(np.std(t_arr, ddof=1))

        # Absolute and Relative Lift
        abs_lift = mean_t - mean_c
        rel_lift_pct = (abs_lift / mean_c * 100) if mean_c != 0 else 0.0

        # 1. Welch's t-test (unequal variances assumed)
        t_stat, p_value_welch = stats.ttest_ind(t_arr, c_arr, equal_var=False)

        # Degrees of freedom for Welch-Satterthwaite
        v_c = (std_c**2) / n_c
        v_t = (std_t**2) / n_t
        df_welch = ((v_c + v_t) ** 2) / ((v_c**2) / (n_c - 1) + (v_t**2) / (n_t - 1)) if (v_c + v_t) > 0 else 1.0

        # Confidence Interval of the difference (95%)
        se_diff = np.sqrt(v_c + v_t)
        t_crit = stats.t.ppf(1 - alpha / 2, df=df_welch)
        ci_lower = abs_lift - t_crit * se_diff
        ci_upper = abs_lift + t_crit * se_diff

        # 2. Mann-Whitney U test (handles heavy Pareto skewness)
        u_stat, p_value_mannwhitney = stats.mannwhitneyu(t_arr, c_arr, alternative="two-sided")

        # 3. Cohen's d effect size (pooled standard deviation)
        pooled_std = np.sqrt(((n_c - 1) * std_c**2 + (n_t - 1) * std_t**2) / (n_c + n_t - 2)) if (n_c + n_t > 2) else 1.0
        cohens_d = (abs_lift / pooled_std) if pooled_std > 0 else 0.0

        is_significant = bool(p_value_welch < alpha)

        if is_significant and abs_lift > 0:
            conclusion = "Statistically Significant Uplift: Variant demonstrates reliable improvement over Control."
        elif is_significant and abs_lift < 0:
            conclusion = "Statistically Significant Decline: Variant significantly underperforms Control."
        else:
            conclusion = "Inconclusive / No Detectable Difference: Observed variance is consistent with random chance."

        return {
            "metric_name": metric_name,
            "sample_sizes": {"control": n_c, "treatment": n_t},
            "control": {"mean": round(mean_c, 2), "std": round(std_c, 2)},
            "treatment": {"mean": round(mean_t, 2), "std": round(std_t, 2)},
            "lift": {
                "absolute_diff": round(abs_lift, 2),
                "relative_lift_pct": round(rel_lift_pct, 2),
                "ci_lower": round(ci_lower, 2),
                "ci_upper": round(ci_upper, 2),
                "confidence_level_pct": round((1 - alpha) * 100, 1),
            },
            "frequentist_tests": {
                "welch_t_stat": round(float(t_stat), 4),
                "welch_p_value": round(float(p_value_welch), 5),
                "mann_whitney_u_stat": round(float(u_stat), 2),
                "mann_whitney_p_value": round(float(p_value_mannwhitney), 5),
                "cohens_d": round(float(cohens_d), 4),
                "is_significant": is_significant,
            },
            "conclusion": conclusion,
        }

    def evaluate_proportion_ab(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        metric_name: str = "Conversion Rate",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Evaluates discrete proportions (e.g., promotional adoption, VIP conversion rate)
        using Two-Proportion z-test and Chi-square contingency analysis.
        """
        if control_total <= 0 or treatment_total <= 0:
            raise ValueError("Group sample totals must be strictly positive integers.")
        if not (0 <= control_successes <= control_total):
            raise ValueError("control_successes must be between 0 and control_total.")
        if not (0 <= treatment_successes <= treatment_total):
            raise ValueError("treatment_successes must be between 0 and treatment_total.")

        p_c = control_successes / control_total
        p_t = treatment_successes / treatment_total

        abs_diff = p_t - p_c
        rel_lift_pct = (abs_diff / p_c * 100) if p_c > 0 else 0.0

        # Pooled proportion for hypothesis testing (H0: p_t = p_c)
        pooled_p = (control_successes + treatment_successes) / (control_total + treatment_total)
        se_pooled = np.sqrt(pooled_p * (1 - pooled_p) * (1 / control_total + 1 / treatment_total))

        # Unpooled standard error for confidence interval
        se_unpooled = np.sqrt(p_c * (1 - p_c) / control_total + p_t * (1 - p_t) / treatment_total)

        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = abs_diff - z_crit * se_unpooled
        ci_upper = abs_diff + z_crit * se_unpooled

        z_stat = (abs_diff / se_pooled) if se_pooled > 0 else 0.0
        p_val_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Chi-Square Test of Independence
        contingency_table = [
            [control_successes, control_total - control_successes],
            [treatment_successes, treatment_total - treatment_successes],
        ]
        chi2_stat, p_val_chi2, dof, _ = stats.chi2_contingency(contingency_table, correction=True)

        is_significant = bool(p_val_z < alpha)

        return {
            "metric_name": metric_name,
            "sample_sizes": {"control": control_total, "treatment": treatment_total},
            "control_conversion_rate": round(p_c, 4),
            "treatment_conversion_rate": round(p_t, 4),
            "lift": {
                "absolute_diff": round(abs_diff, 4),
                "relative_lift_pct": round(rel_lift_pct, 2),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
            },
            "z_test": {"z_statistic": round(float(z_stat), 4), "p_value": round(float(p_val_z), 5)},
            "chi2_test": {"chi2_stat": round(float(chi2_stat), 4), "p_value": round(float(p_val_chi2), 5)},
            "is_significant": is_significant,
        }

    def bayesian_conversion_test(
        self,
        control_conversions: int,
        control_impressions: int,
        treatment_conversions: int,
        treatment_impressions: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_simulations: int = 25000,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes Bayesian Beta-Binomial conjugate updating to compute:
        1. Posterior probability of Treatment beating Control (Probability of Being Best)
        2. Expected Loss (risk of deploying Treatment if Control was actually better)
        3. 95% High Density Credible Intervals on absolute and relative lift
        4. Prescriptive commercial rollout decision
        """
        if control_impressions <= 0 or treatment_impressions <= 0:
            raise ValueError("Impressions must be positive.")
        if not (0 <= control_conversions <= control_impressions):
            raise ValueError("control_conversions must be between 0 and control_impressions.")
        if not (0 <= treatment_conversions <= treatment_impressions):
            raise ValueError("treatment_conversions must be between 0 and treatment_impressions.")

        rng = np.random.default_rng(seed or self.random_state)

        # Posterior Beta parameters
        post_alpha_c = prior_alpha + control_conversions
        post_beta_c = prior_beta + (control_impressions - control_conversions)

        post_alpha_t = prior_alpha + treatment_conversions
        post_beta_t = prior_beta + (treatment_impressions - treatment_conversions)

        # Monte Carlo posterior sampling
        samples_c = rng.beta(post_alpha_c, post_beta_c, size=n_simulations)
        samples_t = rng.beta(post_alpha_t, post_beta_t, size=n_simulations)

        # Probability Treatment > Control
        prob_treatment_beats_control = float(np.mean(samples_t > samples_c))

        # Expected Loss (risk in conversion points)
        # Expected Loss of choosing Treatment = E[max(0, C - T)]
        loss_choose_treatment = float(np.mean(np.maximum(0, samples_c - samples_t)))
        # Expected Loss of choosing Control = E[max(0, T - C)]
        loss_choose_control = float(np.mean(np.maximum(0, samples_t - samples_c)))

        # Absolute and Relative Lift distributions
        diff_samples = samples_t - samples_c
        rel_lift_samples = diff_samples / samples_c

        # 95% Credible Intervals (2.5th and 97.5th percentiles)
        ci_abs = np.percentile(diff_samples, [2.5, 97.5])
        ci_rel = np.percentile(rel_lift_samples, [2.5, 97.5])

        # Decision threshold (Standard industrial Bayesian rule: P > 95% and loss < 0.001)
        if prob_treatment_beats_control >= 0.95 and loss_choose_treatment < 0.001:
            decision = "Deploy Variant (Treatment)"
            decision_rationale = (
                f"Variant has a {prob_treatment_beats_control * 100:.1f}% posterior probability of beating Control "
                f"with negligible expected risk ({loss_choose_treatment * 100:.3f}% conversion risk)."
            )
        elif prob_treatment_beats_control <= 0.05:
            decision = "Retain Control"
            decision_rationale = "Control decisively outperforms Variant; abort experimental rollout."
        else:
            decision = "Continue Experimentation"
            decision_rationale = (
                f"Posterior probability ({prob_treatment_beats_control * 100:.1f}%) has not yet cleared the 95% certainty threshold. "
                "Collect additional transaction volume."
            )

        return {
            "control": {
                "conversions": control_conversions,
                "impressions": control_impressions,
                "posterior_mean": round(float(np.mean(samples_c)), 4),
                "posterior_alpha": post_alpha_c,
                "posterior_beta": post_beta_c,
            },
            "treatment": {
                "conversions": treatment_conversions,
                "impressions": treatment_impressions,
                "posterior_mean": round(float(np.mean(samples_t)), 4),
                "posterior_alpha": post_alpha_t,
                "posterior_beta": post_beta_t,
            },
            "bayesian_metrics": {
                "prob_treatment_better_pct": round(prob_treatment_beats_control * 100, 2),
                "expected_loss_treatment": round(loss_choose_treatment, 5),
                "expected_loss_control": round(loss_choose_control, 5),
                "absolute_lift_mean": round(float(np.mean(diff_samples)), 4),
                "absolute_lift_95_hdi": [round(float(ci_abs[0]), 4), round(float(ci_abs[1]), 4)],
                "relative_lift_mean_pct": round(float(np.mean(rel_lift_samples)) * 100, 2),
                "relative_lift_95_hdi_pct": [
                    round(float(ci_rel[0]) * 100, 2),
                    round(float(ci_rel[1]) * 100, 2),
                ],
            },
            "recommendation": {
                "decision": decision,
                "rationale": decision_rationale,
            },
        }

    def bayesian_revenue_test(
        self,
        control_spend: List[float],
        treatment_spend: List[float],
        n_simulations: int = 25000,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Simulates posterior distribution of mean basket revenue using Bayesian bootstrapping.
        Handles skewed retail spend distributions.
        """
        c_arr = np.asarray(control_spend, dtype=float)
        t_arr = np.asarray(treatment_spend, dtype=float)

        if len(c_arr) < 2 or len(t_arr) < 2:
            raise ValueError("Groups must contain at least 2 observations.")

        rng = np.random.default_rng(seed or self.random_state)

        # Dirichlet Bayesian bootstrap for continuous metric means
        weights_c = rng.dirichlet(np.ones(len(c_arr)), size=n_simulations)
        weights_t = rng.dirichlet(np.ones(len(t_arr)), size=n_simulations)

        posterior_means_c = np.sum(weights_c * c_arr, axis=1)
        posterior_means_t = np.sum(weights_t * t_arr, axis=1)

        diff = posterior_means_t - posterior_means_c
        prob_lift_positive = float(np.mean(diff > 0))

        ci_diff = np.percentile(diff, [2.5, 97.5])
        expected_dollar_gain = float(np.mean(diff))

        return {
            "control_posterior_mean": round(float(np.mean(posterior_means_c)), 2),
            "treatment_posterior_mean": round(float(np.mean(posterior_means_t)), 2),
            "prob_treatment_superior_pct": round(prob_lift_positive * 100, 2),
            "expected_revenue_gain_per_order": round(expected_dollar_gain, 2),
            "credible_interval_95": [round(float(ci_diff[0]), 2), round(float(ci_diff[1]), 2)],
        }

    @staticmethod
    def calculate_sample_size(
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """
        Calculates required sample size per experimental variant using standard power formulation:
        n = 2 * (z_{alpha/2} + z_beta)^2 * p * (1 - p) / (p * mde)^2
        """
        if not (0.0 < baseline_rate < 1.0):
            raise ValueError("baseline_rate must be between 0.0 and 1.0")
        if mde <= 0:
            raise ValueError("mde (Minimum Detectable Effect) must be positive")

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        p = baseline_rate
        var = p * (1 - p)
        delta = p * mde

        n_per_variant = 2 * ((z_alpha + z_beta) ** 2) * var / (delta**2)
        return int(np.ceil(n_per_variant))

    def generate_power_matrix(
        self,
        baseline_rate: float = 0.20,
        mdes: Optional[List[float]] = None,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> List[Dict[str, Any]]:
        """
        Generates a sample size and duration matrix across practical retail Minimum Detectable Effects (MDE).
        """
        if mdes is None:
            mdes = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]

        matrix = []
        for mde in mdes:
            n = self.calculate_sample_size(baseline_rate, mde, alpha=alpha, power=power)
            matrix.append(
                {
                    "baseline_rate_pct": round(baseline_rate * 100, 1),
                    "mde_relative_pct": round(mde * 100, 1),
                    "absolute_target_rate_pct": round((baseline_rate * (1 + mde)) * 100, 2),
                    "sample_size_per_arm": n,
                    "total_experiment_sample_size": n * 2,
                    "target_power_pct": round(power * 100, 1),
                    "significance_alpha_pct": round(alpha * 100, 1),
                }
            )
        return matrix
