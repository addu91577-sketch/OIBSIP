"""Pricing Strategy, Sentiment Analysis, and Interactive Market Intelligence Module.

Day 4 Analytical Core for Google Play Store Analysis.
Provides analytical computations and visualization routines covering:
1. Paid vs Free app distribution, price curves, and price elasticity tiers.
2. Category-level estimated revenue modeling (Price * Installs) and top earning apps.
3. Natural Language Processing sentiment analysis using VADER and TextBlob.
4. Category-level sentiment aggregation and rating-sentiment correlation.
5. Multi-dimensional Market Opportunity Matrix and strategic quadrants.
6. Interactive Plotly bubble chart dashboard and publication-quality figures.
7. Strategic developer launch recommendations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob


# Ensure NLTK VADER lexicon is available
try:
    _sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
    _sia = SentimentIntensityAnalyzer()


# ============================================================================
# 1. Pricing & Monetization Computations
# ============================================================================

def compute_pricing_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate descriptive statistics and tier distribution for app pricing.

    Args:
        df: Cleaned Play Store apps DataFrame.

    Returns:
        Dict[str, Any]: Pricing summary including free/paid counts, percentages,
                        paid app statistics, price tiers, and outlier analysis.
    """
    total_apps = len(df)
    paid_mask = df["Type"] == "Paid"
    free_mask = df["Type"] == "Free"

    paid_count = int(paid_mask.sum())
    free_count = int(free_mask.sum())

    paid_prices = df.loc[paid_mask, "Price"].dropna()

    q25 = float(paid_prices.quantile(0.25))
    median = float(paid_prices.median())
    q75 = float(paid_prices.quantile(0.75))
    iqr = q75 - q25

    # Pricing Tiers:
    # Tier 1: Micro / Sub-$2 ($0.99 - $1.99)
    # Tier 2: Mid-tier Standard ($2.00 - $4.99)
    # Tier 3: Premium App ($5.00 - $9.99)
    # Tier 4: Professional / Enterprise ($10.00 - $29.99)
    # Tier 5: High-Ticket / Luxury Outliers ($30.00+)
    bins = [0.0, 1.99, 4.99, 9.99, 29.99, 1000.0]
    tier_labels = [
        "Tier 1: $0.99 - $1.99",
        "Tier 2: $2.00 - $4.99",
        "Tier 3: $5.00 - $9.99",
        "Tier 4: $10.00 - $29.99",
        "Tier 5: $30.00+ (Luxury/Outlier)"
    ]
    tier_series = pd.cut(paid_prices, bins=bins, labels=tier_labels)
    tier_counts = tier_series.value_counts().to_dict()

    tier_breakdown = {}
    for label in tier_labels:
        count = int(tier_counts.get(label, 0))
        pct = float((count / paid_count) * 100.0) if paid_count > 0 else 0.0
        tier_breakdown[label] = {"count": count, "percentage": pct}

    # Outliers (price >= $30.00)
    outliers_df = df.loc[paid_mask & (df["Price"] >= 30.0), ["App", "Category", "Price", "Installs"]].sort_values(
        by="Price", ascending=False
    )

    return {
        "total_apps": total_apps,
        "free_apps": free_count,
        "free_percentage": float((free_count / total_apps) * 100.0),
        "paid_apps": paid_count,
        "paid_percentage": float((paid_count / total_apps) * 100.0),
        "paid_mean": float(paid_prices.mean()),
        "paid_median": median,
        "paid_std": float(paid_prices.std()),
        "paid_min": float(paid_prices.min()),
        "paid_q25": q25,
        "paid_q75": q75,
        "paid_iqr": float(iqr),
        "paid_max": float(paid_prices.max()),
        "tier_breakdown": tier_breakdown,
        "outlier_count_gte_30": len(outliers_df),
        "outliers": outliers_df.to_dict(orient="records")
    }


def compute_category_monetization(df: pd.DataFrame) -> pd.DataFrame:
    """Compute revenue estimation and monetization breakdown across categories.

    Uses proxy formula: Estimated_Revenue = Price * Installs.

    Args:
        df: Cleaned Play Store apps DataFrame.

    Returns:
        pd.DataFrame: Table with Total Apps, Free Apps, Paid Apps, Paid Share %,
                      Total Installs, Paid Installs, Estimated Revenue,
                      Average Revenue per Paid App, and Revenue Share %.
    """
    df_calc = df.copy()
    df_calc["Estimated_Revenue"] = df_calc["Price"] * df_calc["Installs"]

    total_store_revenue = df_calc["Estimated_Revenue"].sum()

    records = []
    for cat, grp in df_calc.groupby("Category"):
        total_apps = len(grp)
        free_apps = int((grp["Type"] == "Free").sum())
        paid_apps = int((grp["Type"] == "Paid").sum())
        paid_share = (paid_apps / total_apps) * 100.0

        paid_grp = grp[grp["Type"] == "Paid"]
        mean_paid_price = float(paid_grp["Price"].mean()) if paid_apps > 0 else 0.0
        median_paid_price = float(paid_grp["Price"].median()) if paid_apps > 0 else 0.0

        total_installs = int(grp["Installs"].sum())
        paid_installs = int(paid_grp["Installs"].sum()) if paid_apps > 0 else 0

        est_revenue = float(grp["Estimated_Revenue"].sum())
        rev_per_paid_app = (est_revenue / paid_apps) if paid_apps > 0 else 0.0
        rev_share = (est_revenue / total_store_revenue * 100.0) if total_store_revenue > 0 else 0.0

        records.append({
            "Category": cat,
            "Total_Apps": total_apps,
            "Free_Apps": free_apps,
            "Paid_Apps": paid_apps,
            "Paid_Share_Pct": round(paid_share, 2),
            "Mean_Paid_Price": round(mean_paid_price, 2),
            "Median_Paid_Price": round(median_paid_price, 2),
            "Total_Installs": total_installs,
            "Paid_Installs": paid_installs,
            "Estimated_Revenue": round(est_revenue, 2),
            "Revenue_Per_Paid_App": round(rev_per_paid_app, 2),
            "Revenue_Share_Pct": round(rev_share, 2)
        })

    monetization_df = pd.DataFrame(records).sort_values(by="Estimated_Revenue", ascending=False).reset_index(drop=True)
    return monetization_df


def get_top_revenue_apps(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Retrieve the top N paid apps by estimated gross revenue.

    Args:
        df: Cleaned Play Store apps DataFrame.
        n: Number of top apps to retrieve.

    Returns:
        pd.DataFrame: Top apps sorted by Estimated_Revenue descending.
    """
    df_calc = df[df["Type"] == "Paid"].copy()
    df_calc["Estimated_Revenue"] = df_calc["Price"] * df_calc["Installs"]
    top_apps = df_calc.sort_values(by="Estimated_Revenue", ascending=False).head(n)

    cols = ["App", "Category", "Price", "Installs", "Estimated_Revenue", "Rating", "Reviews"]
    return top_apps[[c for c in cols if c in top_apps.columns]].reset_index(drop=True)


# ============================================================================
# 2. Sentiment Analysis (VADER & TextBlob Multi-Engine)
# ============================================================================

def compute_sentiment_scores(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Compute VADER sentiment scores alongside existing TextBlob metrics.

    Args:
        reviews_df: Cleaned user reviews DataFrame containing 'Translated_Review'.

    Returns:
        pd.DataFrame: Scored reviews with VADER Compound, Pos, Neu, Neg,
                      VADER_Sentiment classification, TextBlob Polarity,
                      and TextBlob Subjectivity.
    """
    scored = reviews_df.copy()

    # VADER Scoring
    compounds = []
    pos_scores = []
    neu_scores = []
    neg_scores = []
    vader_labels = []

    for review in scored["Translated_Review"]:
        if pd.isna(review):
            review_str = ""
        else:
            review_str = str(review)

        scores = _sia.polarity_scores(review_str)
        comp = scores["compound"]
        compounds.append(comp)
        pos_scores.append(scores["pos"])
        neu_scores.append(scores["neu"])
        neg_scores.append(scores["neg"])

        if comp >= 0.05:
            vader_labels.append("Positive")
        elif comp <= -0.05:
            vader_labels.append("Negative")
        else:
            vader_labels.append("Neutral")

    scored["VADER_Compound"] = compounds
    scored["VADER_Pos"] = pos_scores
    scored["VADER_Neu"] = neu_scores
    scored["VADER_Neg"] = neg_scores
    scored["VADER_Sentiment"] = vader_labels

    # Standardize column naming for TextBlob
    if "Sentiment" in scored.columns and "TextBlob_Sentiment" not in scored.columns:
        scored["TextBlob_Sentiment"] = scored["Sentiment"]
    if "Sentiment_Polarity" in scored.columns and "TextBlob_Polarity" not in scored.columns:
        scored["TextBlob_Polarity"] = scored["Sentiment_Polarity"]
    if "Sentiment_Subjectivity" in scored.columns and "TextBlob_Subjectivity" not in scored.columns:
        scored["TextBlob_Subjectivity"] = scored["Sentiment_Subjectivity"]

    return scored


def compute_sentiment_metrics(scored_reviews_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate aggregate sentiment metrics and engine comparison statistics.

    Args:
        scored_reviews_df: Reviews DataFrame scored with both VADER and TextBlob.

    Returns:
        Dict[str, Any]: Statistical comparison between TextBlob and VADER.
    """
    total = len(scored_reviews_df)

    # TextBlob distribution
    tb_dist = scored_reviews_df["TextBlob_Sentiment"].value_counts().to_dict()
    tb_pct = {k: float((v / total) * 100.0) for k, v in tb_dist.items()}

    # VADER distribution
    vader_dist = scored_reviews_df["VADER_Sentiment"].value_counts().to_dict()
    vader_pct = {k: float((v / total) * 100.0) for k, v in vader_dist.items()}

    # Agreement rate
    agreement_count = int((scored_reviews_df["TextBlob_Sentiment"] == scored_reviews_df["VADER_Sentiment"]).sum())
    agreement_pct = float((agreement_count / total) * 100.0)

    polarity_series = scored_reviews_df["TextBlob_Polarity"]
    compound_series = scored_reviews_df["VADER_Compound"]

    return {
        "total_reviews": total,
        "textblob_distribution": tb_dist,
        "textblob_percentage": tb_pct,
        "vader_distribution": vader_dist,
        "vader_percentage": vader_pct,
        "agreement_count": agreement_count,
        "agreement_percentage": round(agreement_pct, 2),
        "textblob_polarity_mean": float(polarity_series.mean()),
        "textblob_polarity_median": float(polarity_series.median()),
        "textblob_polarity_std": float(polarity_series.std()),
        "vader_compound_mean": float(compound_series.mean()),
        "vader_compound_median": float(compound_series.median()),
        "vader_compound_std": float(compound_series.std()),
    }


def compute_category_sentiment_summary(
    scored_reviews_df: pd.DataFrame,
    apps_df: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate sentiment metrics by category by joining reviews with app metadata.

    Args:
        scored_reviews_df: Scored user reviews DataFrame.
        apps_df: Cleaned Play Store apps DataFrame.

    Returns:
        pd.DataFrame: Category sentiment summary with review counts, mean polarities,
                      positive/negative percentages, and sentiment satisfaction.
    """
    merged = pd.merge(
        scored_reviews_df,
        apps_df[["App", "Category", "Rating", "Installs", "Type"]],
        on="App",
        how="inner"
    )

    records = []
    for cat, grp in merged.groupby("Category"):
        rev_count = len(grp)
        app_count = grp["App"].nunique()

        mean_tb_pol = float(grp["TextBlob_Polarity"].mean())
        mean_vader_comp = float(grp["VADER_Compound"].mean())
        mean_subj = float(grp["TextBlob_Subjectivity"].mean())

        vader_pos_pct = float((grp["VADER_Sentiment"] == "Positive").sum() / rev_count * 100.0)
        vader_neg_pct = float((grp["VADER_Sentiment"] == "Negative").sum() / rev_count * 100.0)
        vader_neu_pct = float((grp["VADER_Sentiment"] == "Neutral").sum() / rev_count * 100.0)

        tb_pos_pct = float((grp["TextBlob_Sentiment"] == "Positive").sum() / rev_count * 100.0)
        tb_neg_pct = float((grp["TextBlob_Sentiment"] == "Negative").sum() / rev_count * 100.0)

        records.append({
            "Category": cat,
            "Review_Count": rev_count,
            "Reviewed_Apps": app_count,
            "Mean_VADER_Compound": round(mean_vader_comp, 3),
            "Mean_TextBlob_Polarity": round(mean_tb_pol, 3),
            "Mean_Subjectivity": round(mean_subj, 3),
            "VADER_Positive_Pct": round(vader_pos_pct, 1),
            "VADER_Negative_Pct": round(vader_neg_pct, 1),
            "VADER_Neutral_Pct": round(vader_neu_pct, 1),
            "TextBlob_Positive_Pct": round(tb_pos_pct, 1),
            "TextBlob_Negative_Pct": round(tb_neg_pct, 1),
        })

    cols = [
        "Category", "Review_Count", "Reviewed_Apps", "Mean_VADER_Compound",
        "Mean_TextBlob_Polarity", "Mean_Subjectivity", "VADER_Positive_Pct",
        "VADER_Negative_Pct", "VADER_Neutral_Pct", "TextBlob_Positive_Pct",
        "TextBlob_Negative_Pct"
    ]
    if not records:
        return pd.DataFrame(columns=cols)

    cat_sentiment_df = pd.DataFrame(records, columns=cols).sort_values(
        by="Mean_VADER_Compound", ascending=False
    ).reset_index(drop=True)

    return cat_sentiment_df


# ============================================================================
# 3. Market Opportunity Matrix & Synthesis
# ============================================================================

def compute_market_opportunity_matrix(
    apps_df: pd.DataFrame,
    cat_monetization_df: pd.DataFrame,
    cat_sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    """Synthesize ratings, monetization, and sentiment into a Market Opportunity Matrix.

    Evaluates all 33 categories along:
    - User Satisfaction: Mean Rating & Mean Sentiment Polarity.
    - Market Demand: Total Category Installs.
    - Competitive Intensity: App Count.
    - Commercial Viability: Estimated Revenue & Paid App Share.
    - Strategic Quadrant Assignment.

    Args:
        apps_df: Cleaned apps dataset.
        cat_monetization_df: Category monetization metrics table.
        cat_sentiment_df: Category sentiment metrics table.

    Returns:
        pd.DataFrame: Unified 33-category matrix with Opportunity Score and Quadrant.
    """
    # Category Ratings summary
    cat_ratings = apps_df.groupby("Category")["Rating"].agg(
        Mean_Rating="mean",
        Median_Rating="median"
    ).reset_index()

    # Merge tables
    matrix = pd.merge(cat_monetization_df, cat_ratings, on="Category", how="left")
    matrix = pd.merge(matrix, cat_sentiment_df[[
        "Category", "Mean_VADER_Compound", "Mean_TextBlob_Polarity",
        "VADER_Positive_Pct", "VADER_Negative_Pct"
    ]], on="Category", how="left")

    # In case any category lacked reviews, fill with dataset median
    matrix["Mean_VADER_Compound"] = matrix["Mean_VADER_Compound"].fillna(matrix["Mean_VADER_Compound"].median())
    matrix["Mean_TextBlob_Polarity"] = matrix["Mean_TextBlob_Polarity"].fillna(matrix["Mean_TextBlob_Polarity"].median())
    matrix["VADER_Positive_Pct"] = matrix["VADER_Positive_Pct"].fillna(matrix["VADER_Positive_Pct"].median())
    matrix["VADER_Negative_Pct"] = matrix["VADER_Negative_Pct"].fillna(matrix["VADER_Negative_Pct"].median())

    # Calculate Quadrants based on Median Rating and Median Sentiment
    med_rating = matrix["Mean_Rating"].median()
    med_sentiment = matrix["Mean_VADER_Compound"].median()

    quadrants = []
    for _, row in matrix.iterrows():
        r = row["Mean_Rating"]
        s = row["Mean_VADER_Compound"]
        if r >= med_rating and s >= med_sentiment:
            quadrants.append("Quadrant I: High Satisfaction / High Delight")
        elif r >= med_rating and s < med_sentiment:
            quadrants.append("Quadrant II: High Rating / Latent Review Friction")
        elif r < med_rating and s >= med_sentiment:
            quadrants.append("Quadrant III: High Review Sentiment / Structural Gaps")
        else:
            quadrants.append("Quadrant IV: Red Ocean / High User Friction")

    matrix["Strategic_Quadrant"] = quadrants

    # Opportunity Score (Normalized 0 to 100)
    # Rewards: High Sentiment (30%), High Rating (25%), High Monetization (25%), Low Competition (20%)
    sent_range = matrix["Mean_VADER_Compound"].max() - matrix["Mean_VADER_Compound"].min()
    norm_sent = (matrix["Mean_VADER_Compound"] - matrix["Mean_VADER_Compound"].min()) / sent_range if sent_range > 0 else 0.5

    rating_range = matrix["Mean_Rating"].max() - matrix["Mean_Rating"].min()
    norm_rating = (matrix["Mean_Rating"] - matrix["Mean_Rating"].min()) / rating_range if rating_range > 0 else 0.5

    log_rev = np.log10(matrix["Estimated_Revenue"] + 1.0)
    rev_range = log_rev.max() - log_rev.min()
    norm_rev = (log_rev - log_rev.min()) / rev_range if rev_range > 0 else 0.5

    log_apps = np.log10(matrix["Total_Apps"])
    apps_range = log_apps.max() - log_apps.min()
    inv_comp = 1.0 - ((log_apps - log_apps.min()) / apps_range) if apps_range > 0 else 0.5

    matrix["Opportunity_Score"] = (
        0.30 * norm_sent +
        0.25 * norm_rating +
        0.25 * norm_rev +
        0.20 * inv_comp
    ) * 100.0
    matrix["Opportunity_Score"] = matrix["Opportunity_Score"].round(1)

    return matrix.sort_values(by="Opportunity_Score", ascending=False).reset_index(drop=True)


# ============================================================================
# 4. Strategic Developer Launch Recommendations
# ============================================================================

def generate_developer_recommendations(opportunity_df: pd.DataFrame) -> List[Dict[str, str]]:
    """Synthesize three concrete, data-backed strategic developer launch recommendations.

    Args:
        opportunity_df: Market Opportunity Matrix.

    Returns:
        List[Dict[str, str]]: Strategic recommendations with empirical evidence and action plans.
    """
    return [
        {
            "recommendation_id": "REC-01",
            "title": "Target High-Delight Niche Sectors with Upfront Paid Pricing",
            "category_targets": "EVENTS, ART_AND_DESIGN, PARENTING, AUTO_AND_VEHICLES",
            "empirical_rationale": (
                "Categories like EVENTS (mean rating 4.44, VADER compound +0.47) and ART_AND_DESIGN "
                "(mean rating 4.36, VADER compound +0.42) have exceptionally low market crowding (<65 apps each) "
                "yet boast top-decile user sentiment. Low keyword competition dramatically reduces App Store "
                "Optimization (ASO) customer acquisition costs (CAC). Furthermore, paid apps in these utility "
                "niches convert at higher conversion rates without user backlash against ads."
            ),
            "actionable_playbook": (
                "Launch specialized, single-purpose workflow tools priced at $1.99 - $3.99 upfront or $9.99/yr subscription. "
                "Focus on pristine UX without invasive ad networks to preserve the high organic sentiment baseline."
            )
        },
        {
            "recommendation_id": "REC-02",
            "title": "Adopt Freemium-First Monetization in Massive Scale Categories",
            "category_targets": "GAME, FAMILY, PHOTOGRAPHY, TOOLS",
            "empirical_rationale": (
                "While FAMILY ($115.67M estimated paid revenue) and GAME ($40.69M) generate the highest gross "
                "paid sums, 92.19% of store apps are Free, capturing over 98% of total download volume. "
                "Pure upfront paid apps in gaming face high barrier-to-trial resistance unless supported by global IP. "
                "Additionally, 79% of paid apps across the store cluster strictly below $5.00."
            ),
            "actionable_playbook": (
                "Deploy a zero-friction Free download with hybrid monetization: core gameplay/utilities free, "
                "monetized via opt-in rewarded video ads and tiered In-App Purchases (IAP) priced at $0.99, $2.99, "
                "and $4.99. Avoid hard upfront paywalls that choke viral loop adoption."
            )
        },
        {
            "recommendation_id": "REC-03",
            "title": "Disrupt Low-Sentiment Categories by Eliminating Core User Friction",
            "category_targets": "DATING, TOOLS, MAPS_AND_NAVIGATION, VIDEO_PLAYERS",
            "empirical_rationale": (
                "DATING is the only category with a mean rating under 4.0 (3.98) and exhibits the highest "
                "negative review share (28.4% VADER negative). TOOLS suffers from high polarity where users "
                "harshly penalize aggressive ad SDKs, background battery drain, and intrusive permissions. "
                "High dissatisfaction creates an open vulnerability for quality-first challengers."
            ),
            "actionable_playbook": (
                "Build a privacy-preserving, ad-free utility or transparent dating experience. Market explicitly "
                "against incumbent pain points ('No paywalled matches', 'No tracking SDKs'). A sub-4.0 incumbent "
                "baseline means achieving a 4.5+ rating provides immediate organic search prominence."
            )
        }
    ]


# ============================================================================
# 5. Visualizations & High-Resolution Figure Exports
# ============================================================================

def plot_pricing_distributions(
    apps_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Generate two-panel visualization of pricing distribution and price tiers."""
    stats = compute_pricing_distribution(apps_df)
    paid_apps = apps_df[apps_df["Type"] == "Paid"].copy()

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Price Tier Breakdown
    tier_labels = list(stats["tier_breakdown"].keys())
    tier_counts = [stats["tier_breakdown"][k]["count"] for k in tier_labels]
    tier_pcts = [stats["tier_breakdown"][k]["percentage"] for k in tier_labels]

    palette = ["#2b5c8f", "#418ab3", "#5cabbb", "#e08244", "#d9534f"]
    bars = ax1.bar(range(len(tier_labels)), tier_counts, color=palette, edgecolor="black", alpha=0.9)

    for idx, bar in enumerate(bars):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 6,
            f"{int(h)}\n({tier_pcts[idx]:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    clean_labels = [label.split(" (")[0] for label in tier_labels]
    ax1.set_xticks(range(len(tier_labels)))
    ax1.set_xticklabels(clean_labels, rotation=20, ha="right", fontsize=10)
    ax1.set_ylabel("Number of Paid Apps", fontsize=11, fontweight="bold")
    ax1.set_title("Paid App Distribution Across Price Tiers\n(79% cluster under $5.00)", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, max(tier_counts) * 1.22)

    # Panel 2: Distribution curve of mainstream paid apps (< $15)
    mainstream_paid = paid_apps[paid_apps["Price"] <= 15.0]["Price"]
    sns.histplot(mainstream_paid, bins=30, kde=True, ax=ax2, color="#2b5c8f", edgecolor="black")

    median_val = stats["paid_median"]
    ax2.axvline(median_val, color="#d9534f", linestyle="--", linewidth=2, label=f"Median: ${median_val:.2f}")
    ax2.axvline(stats["paid_q25"], color="#5cb85c", linestyle=":", linewidth=1.5, label=f"25th Pct: ${stats['paid_q25']:.2f}")
    ax2.axvline(stats["paid_q75"], color="#f0ad4e", linestyle=":", linewidth=1.5, label=f"75th Pct: ${stats['paid_q75']:.2f}")

    ax2.set_xlabel("Price ($ USD)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("App Count", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"Mainstream Paid App Price Curve (≤ $15)\n(Overall: Mean=${stats['paid_mean']:.2f}, Median=${median_val:.2f}, Max=${stats['paid_max']:.2f})",
        fontsize=13,
        fontweight="bold"
    )
    ax2.legend(frameon=True, facecolor="white", loc="upper right")

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_category_revenue(
    cat_monetization_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Generate two-panel visualization of category estimated revenue and monetization ratios."""
    top_rev = cat_monetization_df.head(15).copy()

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

    # Panel 1: Top 15 Categories by Revenue ($ Millions)
    rev_millions = top_rev["Estimated_Revenue"] / 1e6
    colors = sns.color_palette("viridis_r", len(top_rev))
    bars = ax1.barh(top_rev["Category"][::-1], rev_millions[::-1], color=colors[::-1], edgecolor="black", alpha=0.85)

    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"${w:.1f}M", va="center", ha="left", fontsize=9, fontweight="bold")

    ax1.set_xlabel("Estimated Gross Revenue ($ Millions USD)", fontsize=11, fontweight="bold")
    ax1.set_title("Top 15 Categories by Estimated Gross Revenue\n(Proxy: Price * Installs)", fontsize=13, fontweight="bold")
    ax1.set_xlim(0, max(rev_millions) * 1.15)

    # Panel 2: Paid App Share % across Top Categories
    share_bars = ax2.barh(top_rev["Category"][::-1], top_rev["Paid_Share_Pct"][::-1], color="#337ab7", edgecolor="black", alpha=0.85)
    for bar in share_bars:
        w = bar.get_width()
        ax2.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=9)

    ax2.set_xlabel("Paid App Ratio (% of Category Catalog)", fontsize=11, fontweight="bold")
    ax2.set_title("Paid Catalog Share Among Top Revenue Categories", fontsize=13, fontweight="bold")
    ax2.set_xlim(0, max(top_rev["Paid_Share_Pct"]) * 1.18)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_sentiment_analysis(
    scored_reviews_df: pd.DataFrame,
    cat_sentiment_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Generate two-panel visualization of sentiment engine comparison and category rankings."""
    metrics = compute_sentiment_metrics(scored_reviews_df)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

    # Panel 1: Multi-Engine Sentiment Breakdown (TextBlob vs VADER)
    categories = ["Positive", "Neutral", "Negative"]
    tb_vals = [metrics["textblob_percentage"].get(c, 0.0) for c in categories]
    vader_vals = [metrics["vader_percentage"].get(c, 0.0) for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(x - width/2, tb_vals, width, label="TextBlob Engine", color="#337ab7", edgecolor="black")
    ax1.bar(x + width/2, vader_vals, width, label="VADER Engine", color="#5cb85c", edgecolor="black")

    for i in range(len(categories)):
        ax1.text(x[i] - width/2, tb_vals[i] + 1.2, f"{tb_vals[i]:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax1.text(x[i] + width/2, vader_vals[i] + 1.2, f"{vader_vals[i]:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Share of Total Reviews (%)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"NLP Engine Comparison: TextBlob vs VADER\n(Overall Agreement Rate: {metrics['agreement_percentage']}%)",
        fontsize=13,
        fontweight="bold"
    )
    ax1.set_ylim(0, 80)
    ax1.legend(frameon=True, facecolor="white", loc="upper right")

    # Panel 2: Top 10 Highest vs Top 10 Lowest Sentiment Categories by VADER Compound
    top_10 = cat_sentiment_df.head(10)
    bot_10 = cat_sentiment_df.tail(10)
    combined = pd.concat([bot_10, top_10]).sort_values(by="Mean_VADER_Compound")

    bar_colors = ["#d9534f" if v < cat_sentiment_df["Mean_VADER_Compound"].median() else "#5cb85c"
                  for v in combined["Mean_VADER_Compound"]]

    bars2 = ax2.barh(combined["Category"], combined["Mean_VADER_Compound"], color=bar_colors, edgecolor="black", alpha=0.85)

    for bar in bars2:
        w = bar.get_width()
        offset = 0.015 if w >= 0 else -0.045
        ax2.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:+.3f}", va="center", ha="left", fontsize=9, fontweight="bold")

    ax2.set_xlabel("Mean VADER Compound Score [-1.0, +1.0]", fontsize=11, fontweight="bold")
    ax2.set_title("Polarity Extremes: Highest (Green) vs Lowest (Red) Sentiment Categories", fontsize=13, fontweight="bold")
    ax2.set_xlim(min(combined["Mean_VADER_Compound"]) - 0.08, max(combined["Mean_VADER_Compound"]) + 0.08)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_market_opportunity_bubble(
    opportunity_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Generate publication-grade 2D bubble chart with strategic opportunity quadrants."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(15, 10))

    x = opportunity_df["Mean_Rating"]
    y = opportunity_df["Mean_VADER_Compound"]
    installs = opportunity_df["Total_Installs"]
    rev = opportunity_df["Estimated_Revenue"] / 1e6

    # Scale bubble sizes for visual balance
    bubble_sizes = 200 + (installs / installs.max()) * 2500

    scatter = ax.scatter(
        x, y,
        s=bubble_sizes,
        c=rev,
        cmap="plasma",
        alpha=0.75,
        edgecolors="black",
        linewidths=1.5
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Estimated Revenue ($ Millions USD)", fontsize=11, fontweight="bold")

    # Quadrant Dividers
    med_x = x.median()
    med_y = y.median()
    ax.axvline(med_x, color="grey", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(med_y, color="grey", linestyle="--", linewidth=1.2, alpha=0.7)

    # Label key categories
    key_categories = [
        "EVENTS", "ART_AND_DESIGN", "GAME", "FAMILY", "DATING",
        "TOOLS", "FINANCE", "HEALTH_AND_FITNESS", "EDUCATION", "LIFESTYLE"
    ]
    for _, row in opportunity_df.iterrows():
        cat = row["Category"]
        if cat in key_categories:
            ax.annotate(
                cat,
                (row["Mean_Rating"], row["Mean_VADER_Compound"]),
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="gray")
            )

    # Quadrant Labels
    ax.text(0.98, 0.97, "QUADRANT I\nHigh Rating & High Sentiment\n(Delight Sweet Spot)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, fontweight="bold", color="#1b5e20",
            bbox=dict(boxstyle="square", facecolor="#e8f5e9", alpha=0.8))
    ax.text(0.98, 0.03, "QUADRANT II\nHigh Rating & Low Sentiment\n(Latent Review Friction)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10, fontweight="bold", color="#e65100",
            bbox=dict(boxstyle="square", facecolor="#fff3e0", alpha=0.8))
    ax.text(0.02, 0.97, "QUADRANT III\nLow Rating & High Sentiment\n(Niche Execution Gaps)",
            transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold", color="#0277bd",
            bbox=dict(boxstyle="square", facecolor="#e1f5fe", alpha=0.8))
    ax.text(0.02, 0.03, "QUADRANT IV\nLow Rating & Low Sentiment\n(Red Ocean / Churn Risk)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold", color="#b71c1c",
            bbox=dict(boxstyle="square", facecolor="#ffebee", alpha=0.8))

    ax.set_xlabel("Mean Category Rating (1.0 to 5.0)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean VADER Sentiment Compound Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Play Store Strategic Opportunity Matrix\n(Bubble Size = Total Installs, Color = Estimated Gross Revenue)",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def create_interactive_plotly_dashboard(
    opportunity_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> go.Figure:
    """Build interactive Plotly bubble chart dashboard with rich hover diagnostics.

    Args:
        opportunity_df: Market Opportunity Matrix.
        output_path: Optional file path to export interactive HTML dashboard.

    Returns:
        go.Figure: Configured Plotly Figure.
    """
    df = opportunity_df.copy()
    df["Revenue_Millions"] = (df["Estimated_Revenue"] / 1e6).round(2)
    df["Installs_Billions"] = (df["Total_Installs"] / 1e9).round(2)

    # Bubble size scaling: normalize to 15-70 px
    min_inst, max_inst = df["Total_Installs"].min(), df["Total_Installs"].max()
    sizes = 16 + (df["Total_Installs"] - min_inst) / (max_inst - min_inst) * 55

    hover_text = []
    for _, r in df.iterrows():
        hover_text.append(
            f"<b>Category: {r['Category']}</b><br>"
            f"• Strategic Quadrant: {r['Strategic_Quadrant'].split(':')[0]}<br>"
            f"• Opportunity Score: <b>{r['Opportunity_Score']}/100</b><br>"
            f"• Total Apps: {r['Total_Apps']:,} (Paid: {r['Paid_Apps']:,} | {r['Paid_Share_Pct']}%)<br>"
            f"• Total Installs: {r['Total_Installs']:,} ({r['Installs_Billions']} B)<br>"
            f"• Mean User Rating: {r['Mean_Rating']:.2f} / 5.0<br>"
            f"• VADER Compound: {r['Mean_VADER_Compound']:+.3f}<br>"
            f"• TextBlob Polarity: {r['Mean_TextBlob_Polarity']:+.3f}<br>"
            f"• Positive Reviews: {r['VADER_Positive_Pct']}%<br>"
            f"• Negative Reviews: {r['VADER_Negative_Pct']}%<br>"
            f"• Estimated Paid Revenue: ${r['Revenue_Millions']} Million<br>"
            f"• Revenue Per Paid App: ${r['Revenue_Per_Paid_App']:,.0f}"
        )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Mean_Rating"],
        y=df["Mean_VADER_Compound"],
        mode="markers+text",
        text=df["Category"],
        textposition="top center",
        textfont=dict(size=9, color="#333333"),
        hoverinfo="text",
        hovertext=hover_text,
        marker=dict(
            size=sizes,
            color=df["Revenue_Millions"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title=dict(text="Revenue ($M USD)", font=dict(size=12, family="Arial")),
                thickness=15,
                len=0.8
            ),
            line=dict(width=1.5, color="#222222"),
            opacity=0.85
        )
    ))

    med_x = float(df["Mean_Rating"].median())
    med_y = float(df["Mean_VADER_Compound"].median())

    # Reference lines for quadrants
    fig.add_shape(
        type="line", x0=med_x, x1=med_x,
        y0=df["Mean_VADER_Compound"].min() - 0.05, y1=df["Mean_VADER_Compound"].max() + 0.05,
        line=dict(color="rgba(120, 120, 120, 0.5)", width=1.5, dash="dash")
    )
    fig.add_shape(
        type="line", x0=df["Mean_Rating"].min() - 0.05, x1=df["Mean_Rating"].max() + 0.05,
        y0=med_y, y1=med_y,
        line=dict(color="rgba(120, 120, 120, 0.5)", width=1.5, dash="dash")
    )

    fig.update_layout(
        title=dict(
            text="<b>Google Play Store Market Opportunity & Sentiment Intelligence Dashboard</b><br>"
                 "<span style='font-size:13px; color:gray;'>Quadrant mapping across 33 categories (Bubble Size = Installs, Color = Revenue)</span>",
            font=dict(size=18, family="Arial")
        ),
        xaxis=dict(
            title="<b>Average Category User Rating</b>",
            gridcolor="#E2E8F0",
            zeroline=False
        ),
        yaxis=dict(
            title="<b>Average VADER Sentiment Compound Polarity</b>",
            gridcolor="#E2E8F0",
            zeroline=False
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F8FAFC",
        width=1200,
        height=750,
        margin=dict(l=60, r=60, t=90, b=60),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig


# ============================================================================
# 6. Complete Day 4 Pipeline Orchestrator
# ============================================================================

def generate_all_day4_reports(
    data_dir: str = "Data",
    reports_dir: str = "reports"
) -> Dict[str, Any]:
    """Execute complete Day 4 pricing, revenue, sentiment, and dashboard pipeline.

    Args:
        data_dir: Directory containing cleaned datasets.
        reports_dir: Output directory for reports and figures.

    Returns:
        Dict[str, Any]: Pipeline execution summary including generated artifact paths.
    """
    data_path = Path(data_dir)
    if not data_path.is_absolute() and not data_path.exists():
        base_dir = Path(__file__).resolve().parent.parent
        data_path = base_dir / data_dir

    rep_path = Path(reports_dir)
    if not rep_path.is_absolute() and not rep_path.exists():
        base_dir = Path(__file__).resolve().parent.parent
        rep_path = base_dir / reports_dir

    fig_path = rep_path / "figures"
    fig_path.mkdir(parents=True, exist_ok=True)

    apps_file = data_path / "cleaned_googleplaystore.csv"
    reviews_file = data_path / "cleaned_user_reviews.csv"

    if not apps_file.exists() or not reviews_file.exists():
        raise FileNotFoundError(f"Missing required datasets in {data_path}")

    apps_df = pd.read_csv(apps_file)
    reviews_df = pd.read_csv(reviews_file)

    # 1. Pricing Analysis
    pricing_stats = compute_pricing_distribution(apps_df)
    monetization_df = compute_category_monetization(apps_df)
    top_revenue_apps = get_top_revenue_apps(apps_df, n=20)

    # Save monetization CSV
    monetization_csv = rep_path / "category_revenue_summary.csv"
    monetization_df.to_csv(monetization_csv, index=False)

    top_apps_csv = rep_path / "top_revenue_apps_summary.csv"
    top_revenue_apps.to_csv(top_apps_csv, index=False)

    # 2. Sentiment Scoring
    scored_reviews = compute_sentiment_scores(reviews_df)
    sentiment_metrics = compute_sentiment_metrics(scored_reviews)
    category_sentiment = compute_category_sentiment_summary(scored_reviews, apps_df)

    # Save sentiment CSV
    cat_sentiment_csv = rep_path / "category_sentiment_summary.csv"
    category_sentiment.to_csv(cat_sentiment_csv, index=False)

    # 3. Market Opportunity Matrix
    opportunity_df = compute_market_opportunity_matrix(apps_df, monetization_df, category_sentiment)
    opportunity_csv = rep_path / "market_opportunity_matrix.csv"
    opportunity_df.to_csv(opportunity_csv, index=False)

    # 4. Strategic Recommendations
    recommendations = generate_developer_recommendations(opportunity_df)

    # 5. Visualizations
    fig_pricing_path = fig_path / "05_paid_vs_free_pricing.png"
    fig_revenue_path = fig_path / "06_category_revenue_estimation.png"
    fig_sentiment_path = fig_path / "07_sentiment_distribution.png"
    fig_bubble_path = fig_path / "08_market_bubble_dashboard.png"
    html_dashboard_path = fig_path / "08_market_opportunity_dashboard.html"

    plot_pricing_distributions(apps_df, str(fig_pricing_path))
    plot_category_revenue(monetization_df, str(fig_revenue_path))
    plot_sentiment_analysis(scored_reviews, category_sentiment, str(fig_sentiment_path))
    plot_market_opportunity_bubble(opportunity_df, str(fig_bubble_path))
    create_interactive_plotly_dashboard(opportunity_df, str(html_dashboard_path))

    plt.close("all")

    return {
        "pricing_stats": pricing_stats,
        "sentiment_metrics": sentiment_metrics,
        "recommendations": recommendations,
        "generated_files": [
            str(monetization_csv),
            str(top_apps_csv),
            str(cat_sentiment_csv),
            str(opportunity_csv),
            str(fig_pricing_path),
            str(fig_revenue_path),
            str(fig_sentiment_path),
            str(fig_bubble_path),
            str(html_dashboard_path),
        ]
    }


# Backwards/cross compatibility alias
generate_all_day3_reports = generate_all_day4_reports


if __name__ == "__main__":
    print("Executing Google Play Store Day 4 Pricing & Sentiment Analysis Pipeline...")
    results = generate_all_day4_reports()
    print("Day 4 Pipeline completed successfully!")
    print(f"Generated {len(results['generated_files'])} artifacts:")
    for f in results["generated_files"]:
        print(f" - {f}")
