# 📱 Google Play Store Ecosystem Analysis & Market Intelligence
**Oasis Infobyte Data Analytics Internship · Task Level 2 (Task 4 & Task 5)**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-2.x-150458.svg)
![NumPy](https://img.shields.io/badge/numpy-1.x-013243.svg)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.x-11557c.svg)
![Seaborn](https://img.shields.io/badge/seaborn-0.12+-4c72b0.svg)
![Plotly](https://img.shields.io/badge/plotly-5.x-3F4F75.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.8+-green.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)

---

## 🎯 Objective
Perform a comprehensive data analysis of the Google Play Store ecosystem — cleaning messy real-world data, exploring app categories, analysing ratings and pricing trends, conducting sentiment analysis on user reviews, building interactive business dashboards, and evaluating natural language processing (NLP) algorithms for autocomplete and autocorrect text prediction.

---

## 🛠️ Tech Stack
- **Programming Language:** Python 3.11+
- **Data Manipulation & Analysis:** `pandas`, `numpy`, `scipy`
- **Data Visualization & Exploration:** `matplotlib`, `seaborn`, `plotly`
- **Natural Language Processing:** `nltk`, `textblob`, `pyspellchecker`
- **Interactive Development:** Jupyter Notebook (`.ipynb`)
- **Unit Testing & Verification:** `pytest`
- **Version Control:** Git & GitHub

---

## 📋 Feature Checklist
- [x] **Load the Play Store apps dataset and the user reviews dataset separately**
- [x] **Data cleaning:** fix incorrect data types (`Installs` as string with `+`, `Price` with `$`, `Reviews` as string), handle nulls, remove duplicates
- [x] **Category analysis:** bar chart of app distribution across categories; identify most saturated and niche sectors
- [x] **Ratings analysis:** distribution of app ratings; average rating by category; variance and skewness analysis
- [x] **Size and installs analysis:** scatter plot of app size vs. number of installs; evaluate size-install correlation
- [ ] **Pricing analysis:** paid vs. free app distribution; price distribution for paid apps; category revenue estimation
- [ ] **Sentiment analysis on user reviews:** classify reviews as positive/negative/neutral using TextBlob / VADER
- [ ] **Sentiment by category:** determine which app categories have the most positive and negative user sentiment
- [ ] **Interactive visualization:** develop multidimensional interactive visualization using Plotly
- [ ] **Conclusion:** 3 data-driven strategic insights for developers planning to launch a new app
- [ ] **Task 5 NLP Analytics:** implement and benchmark autocomplete and autocorrect algorithms on real text

---

## 📂 Repository Structure
```text
DataAnalytics-L2-GooglePlayStore/
├── .gitignore                              # Git ignore rules for Python, virtual environments, and Jupyter
├── README.md                               # Project documentation, inspection findings, and architecture
├── requirements.txt                        # Project environment dependencies
├── Data/
│   ├── googleplaystore.csv                 # Raw Google Play Store apps catalog (10,841 rows × 13 columns)
│   ├── googleplaystore_user_reviews.csv    # Raw user sentiment reviews dataset (64,295 rows × 5 columns)
│   ├── cleaned_googleplaystore.csv         # Cleaned, deduplicated apps catalog (9,659 unique apps)
│   ├── cleaned_user_reviews.csv            # Cleaned user reviews dataset (29,692 validated reviews)
│   └── license.txt                         # Dataset citation & license terms
├── Notebooks/
│   ├── 01_playstore_inspection_ingestion.ipynb # Day 1: Raw dataset ingestion & structural schema audit
│   ├── 02_playstore_data_cleaning.ipynb    # Day 2: Production cleaning pipeline & data quality audit
│   └── 03_playstore_category_ratings_size_analysis.ipynb # Day 3: Category, ratings & correlation analysis
├── reports/
│   ├── category_distribution_summary.csv   # Day 3: App volume & percentage share by category
│   ├── category_ratings_summary.csv        # Day 3: Rating metrics (mean, median, IQR) per category
│   ├── data_cleaning_audit_summary.csv     # Day 2: Before vs. after transformation comparison metrics
│   ├── stratified_correlations_summary.csv # Day 3: Size vs installs Spearman correlation per category
│   └── figures/
│       ├── 01_category_distribution.png    # Day 3: Saturated vs niche category distribution
│       ├── 02_rating_distributions.png     # Day 3: Rating KDE, skewness & category boxplots
│       ├── 03_size_vs_installs_scatter.png # Day 3: App size vs log10 installs scatter & trend
│       ├── 04_category_size_installs_correlations.png # Day 3: Stratified category correlation ranking
│       ├── day2_data_cleaning_audit.png    # Day 2: Retention, install tiers, app types & sentiment
│       └── day2_missing_data_profile.png   # Day 2: Missing value rates before vs. after cleaning
├── src/
│   ├── __init__.py                         # Package initialization exposing cleaning and analysis APIs
│   ├── category_ratings_analysis.py        # Day 3: Category, ratings & correlation analytical engine
│   └── data_cleaning.py                    # Day 2: Modular production data cleaning pipeline
└── tests/
    ├── __init__.py                         # Test package initialization
    ├── test_category_ratings.py            # Day 3: Pytest suite for category & correlation calculations
    └── test_cleaning.py                    # Day 2: Pytest validation suite for data transformations
```

---

## 📊 Dataset Schema & Day 1 Inspection Findings

### 1. `googleplaystore.csv` (Apps Catalog)
- **Observations:** 10,841 rows × 13 attributes
- **Schema Attributes:** `App`, `Category`, `Rating`, `Reviews`, `Size`, `Installs`, `Type`, `Price`, `Content Rating`, `Genres`, `Last Updated`, `Current Ver`, `Android Ver`.
- **Key Day 1 Discoveries:**
  - **Corrupted Record Detected:** Row 10472 (`Life Made WI-Fi Touchscreen Photo Frame`) is missing the `Category` value, shifting all subsequent fields leftward (e.g., `Category` contains `1.9`, `Rating` contains `19`, and `Reviews` contains `3.0M`).
  - **Non-Numeric Encodings:**
    - `Reviews` is typed as `object` due to character contamination.
    - `Installs` contains string formatting characters (e.g., `10,000+`, `50,000,000+`, `Free`).
    - `Price` contains currency prefixes (e.g., `$4.99`, `$0.99`, `$0`).
    - `Size` has mixed representations (`19M`, `201k`, and `Varies with device`).
  - **Missing Values:** `Rating` has 1,474 null entries (13.6%), while `Current Ver` and `Android Ver` contain minor null percentages (<0.1%).

### 2. `googleplaystore_user_reviews.csv` (Customer Reviews)
- **Observations:** 64,295 rows × 5 attributes
- **Schema Attributes:** `App`, `Translated_Review`, `Sentiment`, `Sentiment_Polarity`, `Sentiment_Subjectivity`.
- **Key Day 1 Discoveries:**
  - High proportion of unreviewed/null transactions (26,863 rows, ~41.7% contain null reviews).
  - Pre-existing sentiment labels (`Positive`, `Neutral`, `Negative`) provide a baseline benchmark for algorithmic NLP verification.

---

## 🧹 Day 2: Production Data Cleaning & Quality Audit

### 1. Cleaning Architecture & Pipeline Transformations

To transition raw catalog data into a reliable foundation for downstream exploratory data analysis, econometric modeling, and NLP benchmarking, a modular data cleaning pipeline was implemented in [`src/data_cleaning.py`](src/data_cleaning.py):

1. **Shifted / Corrupted Row Remediation:**
   - Isolated and purged row 10472 where an omitted `Category` shifted values into mismatched columns (e.g., `Rating` of 19.0).
2. **Numeric Type Normalization:**
   - **`Reviews`:** Contaminating characters and commas stripped; parsed and safely converted to `int64`.
   - **`Installs`:** Stripped string symbols (`+`, `,`) and cast to `int64` representing true install volume thresholds.
   - **`Price`:** Stripped currency indicators (`$`), coerced invalid values, and cast to `float64` (e.g., `$4.99` $\to 4.99$, `0` $\to 0.0$).
3. **App `Size` String Parsing & Category Imputation:**
   - Implemented `parse_size_to_mb()` converting unit suffixes (`M` $\to 1.0\times$, `k` $\to 1/1024\times$) into continuous megabytes (`float64`).
   - Apps designated as `Varies with device` or missing size were imputed using the **category-specific median size** (`Size_MB_Imputed`), preserving organic category variance (e.g., Games have significantly higher typical sizes than Tools).
4. **App Type Inference & Rating Bounds:**
   - Inferred missing `Type` values from price: records with $\text{Price} > 0$ mapped to `Paid`, and $\text{Price} = 0$ mapped to `Free`.
   - Enforced hard rating domain limits $[1.0, 5.0]$; out-of-bounds ratings set to `NaN`.
   - Standardized `Last Updated` strings into ISO standard `datetime64[ns]`.
5. **Deduplication Strategy (Preserving Most Up-to-Date Records):**
   - The raw catalog contained 1,181 duplicate app entries.
   - Rows were sorted by `Reviews` descending and `Last Updated` descending, then deduplicated by `App` name (`keep='first'`), ensuring that the most mature scrape with the highest engagement metric was preserved.
6. **User Review Dataset Cleansing:**
   - Purged 26,863 null review transactions and 7,740 redundant duplicate reviews.
   - Enforced continuous float boundaries for `Sentiment_Polarity` $[-1.0, 1.0]$ and `Sentiment_Subjectivity` $[0.0, 1.0]$.
   - Resolved missing sentiment labels using calibrated polarity thresholds ($\text{Polarity} > 0.05 \implies \text{Positive}$, $< -0.05 \implies \text{Negative}$, otherwise $\text{Neutral}$).

---

### 2. Before vs. After Transformation Audit

| Metric | Raw Dataset | Cleaned Dataset | Transformation Impact |
| :--- | :---: | :---: | :--- |
| **Apps Record Count** | 10,841 | **9,659** | Removed corrupted row & purged 1,181 duplicate scrape snapshots |
| **Unique App Names** | 9,659 | **9,659** | **100% unique applications** (zero identity duplication) |
| **Corrupted Shifted Rows** | 1 | **0** | Row 10472 safely excised |
| **Reviews Data Type** | `object` (string with commas) | **`int64`** | Numeric conversion enabling direct aggregation and ranking |
| **Installs Data Type** | `object` (`+` and `,` formatted) | **`int64`** | Numeric thresholds from $0$ to $1,000,000,000+$ |
| **Price Data Type** | `object` (`$` formatted string) | **`float64`** | Numeric currency value ($0.00 to $400.00) |
| **Size Representation** | `object` (`M`, `k`, strings) | **`float64`** | Continuous megabytes (`Size_MB`) & category-imputed (`Size_MB_Imputed`) |
| **Rating Domain Violations** | 1 (`Rating = 19.0`) | **0** | Bounded strictly to $[1.0, 5.0]$ |
| **User Reviews Total Count** | 64,295 | **29,692** | Purged 26,863 null reviews & 7,740 exact duplicate comments |
| **Review Missing Translations** | 26,863 (41.8%) | **0 (0.0%)** | 100% complete, authentic user review text |

---

### 3. Visual Quality Profiling

#### A. Missing Value Rate Comparison
![Missing Value Profiling](reports/figures/day2_missing_data_profile.png)

#### B. Retention, Install Tiering & Sentiment Distribution
![Data Cleaning Audit](reports/figures/day2_data_cleaning_audit.png)

---

## 📈 Day 3: Category, Ratings & Size Market Analysis

Day 3 executes comprehensive exploratory data analysis, market saturation profiling, rating skewness characterization, and app size vs install econometric modeling across the 9,659 unique apps catalog via [`src/category_ratings_analysis.py`](src/category_ratings_analysis.py) and [`Notebooks/03_playstore_category_ratings_size_analysis.ipynb`](Notebooks/03_playstore_category_ratings_size_analysis.ipynb).

---

### 1. Category Distribution & Market Saturation Analysis
Across 33 distinct app categories in the Google Play Store:
- **Hyper-Saturated Sectors**:
  - `FAMILY`: **1,874 apps** (19.40% catalog share)
  - `GAME`: **946 apps** (9.79% catalog share)
  - `TOOLS`: **829 apps** (8.58% catalog share)
  - *Strategic Takeaway*: The top 3 categories account for **37.78%** of the entire catalog, while the top 5 represent nearly half (**46.22%**). New entrants in these domains encounter hyper-competitive keyword bidding and entrenched incumbents with tens of millions of installs.
- **Underserved Niche Opportunities**:
  - `BEAUTY` (53 apps, 0.55%), `COMICS` (56 apps, 0.58%), `PARENTING` (60 apps, 0.62%), `ART_AND_DESIGN` (61 apps, 0.63%), and `EVENTS` (64 apps, 0.66%).
  - *Strategic Advantage*: Low App Store Optimization (ASO) barrier to entry with highly focused target audiences and minimal incumbent dominance.

| Category Rank | Category Name | App Count | Market Share (%) | Cumulative Share (%) | Competitive Status |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `FAMILY` | 1,874 | 19.40% | 19.40% | Hyper-Saturated |
| **2** | `GAME` | 946 | 9.79% | 29.20% | Hyper-Saturated |
| **3** | `TOOLS` | 829 | 8.58% | 37.78% | Hyper-Saturated |
| **4** | `BUSINESS` | 420 | 4.35% | 42.13% | Moderate Saturation |
| **5** | `MEDICAL` | 395 | 4.09% | 46.22% | Moderate Saturation |
| ... | ... | ... | ... | ... | ... |
| **29** | `EVENTS` | 64 | 0.66% | 97.62% | Niche Opportunity |
| **30** | `ART_AND_DESIGN` | 61 | 0.63% | 98.25% | Niche Opportunity |
| **31** | `PARENTING` | 60 | 0.62% | 98.87% | Niche Opportunity |
| **32** | `COMICS` | 56 | 0.58% | 99.45% | Niche Opportunity |
| **33** | `BEAUTY` | 53 | 0.55% | 100.00% | Niche Opportunity |

#### Visual Distribution: Saturated vs. Niche Sectors
![Category Distribution](reports/figures/01_category_distribution.png)

---

### 2. Rating Distribution & Category Performance

- **Statistical Characteristics of User Ratings**:
  - **Sample Coverage**: 8,196 rated apps (84.85% coverage; 1,463 unrated apps).
  - **Mean Rating**: **4.173** | **Median Rating**: **4.300** | **Std Dev**: **0.536**.
  - **Interquartile Range (IQR)**: $[4.00, 4.50]$ ($\Delta = 0.50$).
  - **Skewness**: **-1.745** (Strong systemic negative skew / rating inflation).
  - *Market Implication*: With 75% of apps scoring $\ge 4.0$, any app rated below **4.0** sits in the bottom quartile of the marketplace and suffers severe algorithmic ranking suppression and conversion loss.
- **Top 5 Highest Rated Categories**:
  1. `EVENTS`: Mean **4.436** (Median: 4.5, Coverage: 70.3%)
  2. `ART_AND_DESIGN`: Mean **4.359** (Median: 4.4, Coverage: 96.7%)
  3. `EDUCATION`: Mean **4.355** (Median: 4.4, Std: **0.268** — most consistent quality in the store)
  4. `BOOKS_AND_REFERENCE`: Mean **4.345** (Median: 4.5, Coverage: 79.7%)
  5. `PERSONALIZATION`: Mean **4.332** (Median: 4.4, Coverage: 76.8%)
- **Bottom 5 Lowest Rated Categories**:
  1. `DATING`: Mean **3.980** (Median: 4.1, Std: 0.629 — *only category in the store with mean < 4.0*)
  2. `MAPS_AND_NAVIGATION`: Mean **4.036** (Median: 4.2, Std: 0.527)
  3. `TOOLS`: Mean **4.040** (Median: 4.2, Std: 0.619 — extreme polarity between reliable utilities and ad-heavy clones)
  4. `VIDEO_PLAYERS`: Mean **4.045** (Median: 4.2, Std: 0.562)
  5. `TRAVEL_AND_LOCAL`: Mean **4.070** (Median: 4.2, Std: 0.532)

#### Visual Distribution: Rating KDE & Cross-Category Boxplots
![Rating Distribution](reports/figures/02_rating_distributions.png)

---

### 3. App Size vs. Number of Installs Correlation Analysis

- **Hypothesis Evaluation**:
  - *Naive Assumption*: Users avoid large downloads due to cellular data caps and storage constraints, implying a negative correlation between binary size and installs.
  - *Empirical Finding*: **Statistically significant positive correlation** between app binary size and install volume!
- **Correlation Metrics ($n = 8,431$ apps with explicit sizes)**:
  - **Pearson Linear Correlation ($r$)**: $+0.1342$ ($p = 3.46 \times 10^{-35}$)
  - **Spearman Rank Correlation ($\rho$)**: **$+0.3101$** ($p = 2.13 \times 10^{-187}$)
  - **Pearson with $\log_{10}(\text{Installs})$**: $+0.2998$ ($p = 1.26 \times 10^{-174}$)
- **The "Feature Richness" Paradox**:
  - Applications achieving 10M+, 100M+, and 1B+ installs invest extensively in high-fidelity graphics, localization assets, multi-architecture native binaries, and comprehensive feature sets.
  - In modern cellular and high-speed Wi-Fi ecosystems, users comfortably tolerate larger downloads in exchange for polished, feature-complete experiences.

#### Visual Correlation: Size vs. Installs Scatter & Log-Linear Trend
![Size vs Installs Scatter](reports/figures/03_size_vs_installs_scatter.png)

---

### 4. Stratified Category-Level Correlations
The size-to-installs dependency varies substantially by application domain:
- **Strongest Positive Dependency**: `SHOPPING` ($\rho = +0.578$), `ART_AND_DESIGN` ($\rho = +0.552$), `PHOTOGRAPHY` ($\rho = +0.459$), `FAMILY` ($\rho = +0.450$), `GAME` ($\rho = +0.349$).
- **Neutral / Negligible Dependency**: `PRODUCTIVITY` ($\rho = +0.001$), `MEDICAL` ($\rho = -0.004$), `PERSONALIZATION` ($\rho = -0.016$), `TOOLS` ($\rho = +0.085$). In functional utilities, users prioritize lightweight, instant execution over feature bloat.

#### Visual Correlation: Stratified Spearman Coefficients by Sector
![Stratified Category Correlations](reports/figures/04_category_size_installs_correlations.png)

---

### 5. Automated Testing & Verification

A dedicated unit and regression test suite is implemented across [`tests/test_cleaning.py`](tests/test_cleaning.py) and [`tests/test_category_ratings.py`](tests/test_category_ratings.py):
- **`test_compute_category_distribution`:** Validates category aggregation across all 33 categories, 100% cumulative share bounds, and identifies top (`FAMILY`) and bottom (`BEAUTY`) sectors.
- **`test_compute_rating_statistics`:** Verifies 8,196 rated vs 1,463 unrated counts, left-skewed rating inflation ($\text{skew} < -1.0$), median (4.30), and IQR bounds.
- **`test_compute_category_ratings`:** Validates category-level mean rating sorting (`EVENTS` highest, `DATING` lowest < 4.0), and coverage percentage metrics.
- **`test_compute_size_installs_correlation`:** Verifies Spearman rank correlation ($\rho > 0.25$, $p < 10^{-50}$) and log-transformed Pearson correlation.
- **`test_compute_stratified_correlations`:** Validates domain stratification for categories with $n \ge 30$ and checks top correlation sectors (`SHOPPING`, `ART_AND_DESIGN`).
- **`test_generate_category_ratings_reports`:** Confirms end-to-end report and figure serialization integrity using mock/temporary directories.

```bash
$ pytest tests/ -v
============================= test session starts =============================
platform win32 -- Python 3.14.8, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\Iba Zaidi\Desktop\OIBSIP\DataAnalytics-L2-GooglePlayStore
collected 10 items

tests/test_category_ratings.py::test_compute_category_distribution PASSED [ 10%]
tests/test_category_ratings.py::test_compute_rating_statistics PASSED    [ 20%]
tests/test_category_ratings.py::test_compute_category_ratings PASSED     [ 30%]
tests/test_category_ratings.py::test_compute_size_installs_correlation PASSED [ 40%]
tests/test_category_ratings.py::test_compute_stratified_correlations PASSED [ 50%]
tests/test_category_ratings.py::test_generate_category_ratings_reports PASSED [ 60%]
tests/test_cleaning.py::test_parse_size_to_mb PASSED                     [ 70%]
tests/test_cleaning.py::test_clean_playstore_apps PASSED                 [ 80%]
tests/test_cleaning.py::test_clean_user_reviews PASSED                   [ 90%]
tests/test_cleaning.py::test_saved_cleaned_datasets_exist PASSED         [100%]

====================== 10 passed in 18.09s =======================
```

---

## 🚀 Execution & Reproduction Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute Data Cleaning Pipeline (Day 2)
```bash
python src/data_cleaning.py
```

### 3. Execute Category, Ratings & Size Analysis Pipeline (Day 3)
```bash
python src/category_ratings_analysis.py
```

### 4. Run Automated Pytest Suite
```bash
pytest tests/ -v
```

### 5. Launch Interactive Jupyter Notebooks
```bash
jupyter notebook Notebooks/01_playstore_inspection_ingestion.ipynb
jupyter notebook Notebooks/02_playstore_data_cleaning.ipynb
jupyter notebook Notebooks/03_playstore_category_ratings_size_analysis.ipynb
```
