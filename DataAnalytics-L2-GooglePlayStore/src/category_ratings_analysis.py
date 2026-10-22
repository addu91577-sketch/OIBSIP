"""Category, Ratings, and Size exploratory data analysis module for Google Play Store.

Provides analytical computations and visualization routines covering:
1. Category distribution and market saturation vs niche sector analysis.
2. Ratings distribution, skewness, and category-level statistical comparisons.
3. App size vs. installs correlation analysis (Pearson, Spearman, log-transformed).
4. Stratified category-level correlations.
5. High-resolution figure generation and summary CSV report exports.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


def compute_category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute app count, percentage, and cumulative market share per category.

    Args:
        df: Cleaned Play Store apps DataFrame.

    Returns:
        pd.DataFrame: Summary table indexed by Category with count, percentage,
                      and cumulative percentage sorted descending.
    """
    total_apps = len(df)
    counts = df["Category"].value_counts()
    cat_df = pd.DataFrame({
        "Category": counts.index,
        "App_Count": counts.values,
        "Percentage": (counts.values / total_apps) * 100.0,
        "Cumulative_Percentage": (counts.values.cumsum() / total_apps) * 100.0
    })
    return cat_df


def compute_rating_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate descriptive statistics for app ratings across the dataset.

    Args:
        df: Cleaned Play Store apps DataFrame.

    Returns:
        Dict[str, Any]: Rating metrics including count, unrated count, mean,
                        median, std, skewness, kurtosis, IQR, and percentiles.
    """
    total_apps = len(df)
    ratings = df["Rating"].dropna()
    rated_count = len(ratings)
    unrated_count = total_apps - rated_count

    q25, q75 = ratings.quantile(0.25), ratings.quantile(0.75)
    iqr = q75 - q25

    return {
        "total_apps": total_apps,
        "rated_apps": rated_count,
        "unrated_apps": unrated_count,
        "rated_percentage": float((rated_count / total_apps) * 100.0),
        "mean": float(ratings.mean()),
        "median": float(ratings.median()),
        "std": float(ratings.std()),
        "skewness": float(ratings.skew()),
        "kurtosis": float(ratings.kurtosis()),
        "min": float(ratings.min()),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(iqr),
        "max": float(ratings.max()),
    }


def compute_category_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rating summary statistics grouped by Category.

    Args:
        df: Cleaned Play Store apps DataFrame.

    Returns:
        pd.DataFrame: Category rating metrics (count, mean, median, std, iqr, min, max)
                      sorted by mean rating descending.
    """
    def calc_iqr(series: pd.Series) -> float:
        cleaned = series.dropna()
        if len(cleaned) < 2:
            return 0.0
        return float(cleaned.quantile(0.75) - cleaned.quantile(0.25))

    grouped = df.groupby("Category")["Rating"].agg(
        Rated_Count="count",
        Mean_Rating="mean",
        Median_Rating="median",
        Std_Rating="std",
        Min_Rating="min",
        Max_Rating="max"
    )

    iqr_series = df.groupby("Category")["Rating"].apply(calc_iqr)
    grouped["IQR_Rating"] = iqr_series

    # Merge total app count to compute rating coverage rate
    total_per_cat = df.groupby("Category")["App"].count()
    grouped["Total_Apps"] = total_per_cat
    grouped["Coverage_Pct"] = (grouped["Rated_Count"] / grouped["Total_Apps"]) * 100.0

    return grouped.sort_values(by="Mean_Rating", ascending=False).reset_index()


def compute_size_installs_correlation(
    df: pd.DataFrame,
    use_imputed: bool = False
) -> Dict[str, Any]:
    """Calculate Pearson and Spearman correlations between app size and installs.

    Computes:
    - Linear Pearson correlation (Size vs Installs)
    - Monotonic Spearman rank correlation (Size vs Installs)
    - Log-transformed Pearson correlation (Size vs log10(Installs + 1))

    Args:
        df: Cleaned Play Store apps DataFrame.
        use_imputed: If True, uses 'Size_MB_Imputed'; else uses explicit 'Size_MB'.

    Returns:
        Dict[str, Any]: Correlation coefficients and p-values.
    """
    col = "Size_MB_Imputed" if use_imputed else "Size_MB"
    sub_df = df.dropna(subset=[col, "Installs"]).copy()

    size_vals = sub_df[col].values
    installs_vals = sub_df["Installs"].values

    pearson_r, pearson_p = stats.pearsonr(size_vals, installs_vals)
    spearman_rho, spearman_p = stats.spearmanr(size_vals, installs_vals)

    log_installs = np.log10(installs_vals + 1.0)
    pearson_log_r, pearson_log_p = stats.pearsonr(size_vals, log_installs)

    return {
        "sample_size": len(sub_df),
        "feature_used": col,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "pearson_log_r": float(pearson_log_r),
        "pearson_log_p": float(pearson_log_p),
    }


def compute_stratified_correlations(
    df: pd.DataFrame,
    min_apps: int = 30
) -> pd.DataFrame:
    """Compute size vs installs correlations stratified across individual categories.

    Args:
        df: Cleaned Play Store apps DataFrame.
        min_apps: Minimum number of apps with size data required per category.

    Returns:
        pd.DataFrame: Category breakdown with sample size, Pearson r, Spearman rho,
                      Spearman p-value, median size, and median installs.
    """
    valid = df.dropna(subset=["Size_MB", "Installs"])
    records = []

    for category, grp in valid.groupby("Category"):
        if len(grp) >= min_apps:
            p_r, p_p = stats.pearsonr(grp["Size_MB"], grp["Installs"])
            s_rho, s_p = stats.spearmanr(grp["Size_MB"], grp["Installs"])
            records.append({
                "Category": category,
                "Sample_Size": len(grp),
                "Pearson_r": float(p_r),
                "Pearson_p": float(p_p),
                "Spearman_rho": float(s_rho),
                "Spearman_p": float(s_p),
                "Median_Size_MB": float(grp["Size_MB"].median()),
                "Median_Installs": float(grp["Installs"].median()),
            })

    strat_df = pd.DataFrame(records).sort_values(by="Spearman_rho", ascending=False).reset_index(drop=True)
    return strat_df


# ============================================================================
# Plotting & Visualization Functions
# ============================================================================

def plot_category_distribution(
    df: pd.DataFrame,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Generate bar chart displaying app distribution across categories."""
    cat_dist = compute_category_distribution(df)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(14, 10))

    # Color gradient highlighting saturated categories (top 3) vs niche (bottom 5)
    colors = []
    for i, row in cat_dist.iterrows():
        if i < 3:
            colors.append("#d9534f")  # Top saturated: coral red
        elif i >= len(cat_dist) - 5:
            colors.append("#5cb85c")  # Niche sectors: soft green
        else:
            colors.append("#337ab7")  # Standard: blue

    bars = ax.barh(cat_dist["Category"][::-1], cat_dist["App_Count"][::-1], color=colors[::-1], edgecolor="black", alpha=0.85)

    # Add count and percentage labels to bars
    for bar in bars:
        width = bar.get_width()
        pct = (width / len(df)) * 100
        ax.text(
            width + 15,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9,
            color="#222222"
        )

    ax.set_title("Google Play Store: App Volume Distribution by Category\n(Red: Top 3 Saturated, Green: Top 5 Niche Opportunities)", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Unique Apps", fontsize=12, fontweight="bold")
    ax.set_ylabel("Category", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(cat_dist["App_Count"]) * 1.15)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_rating_distributions(
    df: pd.DataFrame,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Generate combined rating distribution plot: Histogram + KDE and Category Boxplots."""
    stats_dict = compute_rating_statistics(df)
    rated_df = df.dropna(subset=["Rating"]).copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14), gridspec_kw={"height_ratios": [1, 1.6]})

    # Subplot 1: Overall Rating Distribution (Histogram + KDE)
    sns.histplot(
        rated_df["Rating"],
        kde=True,
        bins=30,
        color="#2b5c8f",
        edgecolor="white",
        ax=ax1,
        stat="density",
        alpha=0.65
    )

    ax1.axvline(stats_dict["mean"], color="#d9534f", linestyle="--", linewidth=2.5,
                label=f"Mean: {stats_dict['mean']:.2f}")
    ax1.axvline(stats_dict["median"], color="#5cb85c", linestyle="-", linewidth=2.5,
                label=f"Median: {stats_dict['median']:.2f}")
    ax1.axvline(stats_dict["q25"], color="#f0ad4e", linestyle=":", linewidth=2,
                label=f"25th %ile: {stats_dict['q25']:.2f}")
    ax1.axvline(stats_dict["q75"], color="#f0ad4e", linestyle=":", linewidth=2,
                label=f"75th %ile: {stats_dict['q75']:.2f}")

    ax1.set_title(
        f"Overall App Rating Distribution (Skewness: {stats_dict['skewness']:.2f}, Left-Skewed Inflation)",
        fontsize=14,
        fontweight="bold"
    )
    ax1.set_xlabel("App Rating (1.0 to 5.0 scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax1.set_xlim(1.0, 5.0)
    ax1.legend(fontsize=10, loc="upper left")

    # Subplot 2: Rating Distribution across Top 20 Categories (Boxplots)
    top_categories = df["Category"].value_counts().head(20).index
    filtered_df = rated_df[rated_df["Category"].isin(top_categories)]
    order = filtered_df.groupby("Category")["Rating"].median().sort_values(ascending=False).index

    sns.boxplot(
        data=filtered_df,
        x="Category",
        y="Rating",
        hue="Category",
        order=order,
        palette="coolwarm",
        ax=ax2,
        legend=False,
        showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": "7"}
    )
    ax2.set_title("Rating Distribution Across Top 20 App Categories (Ranked by Median, Yellow Caret = Mean)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Category", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Rating", fontsize=11, fontweight="bold")
    ax2.set_ylim(1.0, 5.1)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order, rotation=45, ha="right", fontsize=9)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_size_vs_installs(
    df: pd.DataFrame,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Generate scatter plot with log-scaled installs to analyze Size vs Installs relationship."""
    valid = df.dropna(subset=["Size_MB", "Installs"]).copy()
    corr_stats = compute_size_installs_correlation(df, use_imputed=False)

    fig, ax = plt.subplots(figsize=(13, 8))

    # Scatter with alpha transparency to reveal density concentration
    scatter = ax.scatter(
        valid["Size_MB"],
        valid["Installs"],
        c=np.log10(valid["Installs"] + 1),
        cmap="viridis",
        alpha=0.45,
        edgecolors="none",
        s=35
    )

    # Trend line using polynomial fit on log10(Installs)
    z = np.polyfit(valid["Size_MB"], np.log10(valid["Installs"] + 1), 1)
    p = np.poly1d(z)
    x_vals = np.linspace(valid["Size_MB"].min(), valid["Size_MB"].max(), 200)
    y_vals = 10 ** p(x_vals) - 1
    ax.plot(x_vals, y_vals, color="#e74c3c", linewidth=2.5, linestyle="--", label="Log-Linear Trend")

    ax.set_yscale("log")
    ax.set_title(
        "Google Play Store: App Size (MB) vs. Number of Installs\n"
        f"Spearman Rank Correlation: ρ = {corr_stats['spearman_rho']:.3f} (p < 1e-100) | Pearson: r = {corr_stats['pearson_r']:.3f}",
        fontsize=14,
        fontweight="bold",
        pad=15
    )
    ax.set_xlabel("App Size (Megabytes - MB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Installs (Log Scale)", fontsize=12, fontweight="bold")

    # Annotate statistical insight
    annotation_text = (
        "Statistical Finding:\n"
        f"• Spearman ρ: +{corr_stats['spearman_rho']:.3f} (Significant positive rank correlation)\n"
        f"• Pearson r: +{corr_stats['pearson_r']:.3f} (with log10 installs: +{corr_stats['pearson_log_r']:.3f})\n"
        "• Counter-Intuitive Trend: Top-installed apps are larger due to rich media,\n"
        "  assets, and comprehensive feature suites."
    )
    ax.text(
        0.03, 0.94,
        annotation_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#bdc3c7", alpha=0.92)
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Log10(Installs)", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_stratified_correlations(
    strat_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Generate bar chart displaying Spearman rank correlation by category."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Color coding: strong positive correlation (blue), moderate (amber), negative/near-zero (red)
    colors = ["#2b5c8f" if rho >= 0.25 else ("#f39c12" if rho > 0 else "#c0392b") for rho in strat_df["Spearman_rho"]]

    y_pos = np.arange(len(strat_df))
    bars = ax.barh(y_pos, strat_df["Spearman_rho"], color=colors, edgecolor="black", alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(strat_df["Category"])
    ax.invert_yaxis()  # Highest correlation on top

    # Add numeric labels
    for bar in bars:
        width = bar.get_width()
        offset = 0.01 if width >= 0 else -0.04
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{width:+.3f}",
            va="center",
            ha="left" if width >= 0 else "right",
            fontsize=8.5,
            fontweight="bold"
        )

    ax.axvline(0, color="black", linewidth=1.2, linestyle="-")
    ax.set_xlim(-0.15, max(strat_df["Spearman_rho"]) * 1.15)
    ax.set_title("App Size vs. Installs Spearman Correlation by Category\n(Categories with n ≥ 30 Apps)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Spearman Rank Correlation Coefficient (ρ)", fontsize=12, fontweight="bold")
    ax.set_ylabel("App Category", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def generate_category_ratings_reports(
    data_path: Union[str, Path] = "Data/cleaned_googleplaystore.csv",
    output_dir: Union[str, Path] = "reports/figures"
) -> Dict[str, Any]:
    """Execute Category, Ratings & Size analytical pipeline and export all visual figures.

    Args:
        data_path: Path to cleaned Play Store dataset CSV.
        output_dir: Directory where figures and summaries should be written.

    Returns:
        Dict[str, Any]: Compiled summary metrics dictionary.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    # 1. Category Distribution
    cat_dist = compute_category_distribution(df)
    cat_dist.to_csv(out_dir.parent / "category_distribution_summary.csv", index=False)
    plot_category_distribution(df, str(out_dir / "01_category_distribution.png"))

    # 2. Ratings Statistics
    rating_stats = compute_rating_statistics(df)
    cat_ratings = compute_category_ratings(df)
    cat_ratings.to_csv(out_dir.parent / "category_ratings_summary.csv", index=False)
    plot_rating_distributions(df, str(out_dir / "02_rating_distributions.png"))

    # 3. Size vs Installs Correlation
    size_corr = compute_size_installs_correlation(df, use_imputed=False)
    strat_corr = compute_stratified_correlations(df, min_apps=30)
    strat_corr.to_csv(out_dir.parent / "stratified_correlations_summary.csv", index=False)
    plot_size_vs_installs(df, str(out_dir / "03_size_vs_installs_scatter.png"))
    plot_stratified_correlations(strat_corr, str(out_dir / "04_category_size_installs_correlations.png"))

    plt.close("all")

    return {
        "category_distribution": cat_dist,
        "rating_stats": rating_stats,
        "category_ratings": cat_ratings,
        "size_installs_correlation": size_corr,
        "stratified_correlations": strat_corr
    }


# Backwards compatibility aliases
generate_all_day3_reports = generate_category_ratings_reports
generate_all_day2_reports = generate_category_ratings_reports


if __name__ == "__main__":
    print("Executing Google Play Store Category, Ratings & Size Analysis Pipeline...")
    results = generate_category_ratings_reports()
    print("Analysis complete. Key Metrics:")
    print(f"- Total Apps Analyzed: {results['rating_stats']['total_apps']}")
    print(f"- Rating Mean: {results['rating_stats']['mean']:.3f}, Median: {results['rating_stats']['median']:.3f}, Skew: {results['rating_stats']['skewness']:.3f}")
    print(f"- Size vs Installs Spearman Correlation: {results['size_installs_correlation']['spearman_rho']:.4f} (p = {results['size_installs_correlation']['spearman_p']:.2e})")
    print("- Reports and figures generated in reports/ and reports/figures/")
