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
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Data Visualization & Exploration:** `matplotlib`, `seaborn`, `plotly`
- **Natural Language Processing:** `nltk`, `textblob`, `pyspellchecker`
- **Interactive Development:** Jupyter Notebook (`.ipynb`)
- **Unit Testing & Verification:** `pytest`
- **Version Control:** Git & GitHub

---

## 📋 Feature Checklist
- [x] **Load the Play Store apps dataset and the user reviews dataset separately**
- [x] **Data cleaning:** fix incorrect data types (`Installs` as string with `+`, `Price` with `$`, `Reviews` as string), handle nulls, remove duplicates
- [ ] **Category analysis:** bar chart of app distribution across categories; identify most saturated and niche sectors
- [ ] **Ratings analysis:** distribution of app ratings; average rating by category; variance and skewness analysis
- [ ] **Size and installs analysis:** scatter plot of app size vs. number of installs; evaluate size-install correlation
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
│   └── 02_playstore_data_cleaning.ipynb    # Day 2: Production cleaning pipeline & data quality audit
├── reports/
│   ├── data_cleaning_audit_summary.csv     # Before vs. after transformation comparison metrics
│   └── figures/
│       ├── day2_missing_data_profile.png   # Missing value rates before vs. after cleaning
│       └── day2_data_cleaning_audit.png    # Retention, install tiers, app types & sentiment
├── src/
│   ├── __init__.py                         # Package initialization exposing cleaning APIs
│   └── data_cleaning.py                    # Modular production data cleaning pipeline
└── tests/
    ├── __init__.py                         # Test package initialization
    └── test_cleaning.py                    # Pytest validation suite for data transformations
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
Comparison of attribute null rates before and after pipeline execution:
![Missing Value Profiling](reports/figures/day2_missing_data_profile.png)

#### B. Retention, Install Tiering & Sentiment Distribution
Comprehensive 4-panel visual audit verifying app deduplication, install scale distribution, monetization breakdown, and user sentiment class balance:
![Data Cleaning Audit](reports/figures/day2_data_cleaning_audit.png)

---

### 4. Automated Testing & Verification

A dedicated unit and regression test suite is implemented in [`tests/test_cleaning.py`](tests/test_cleaning.py) and executed via `pytest`:
- **`test_parse_size_to_mb`:** Validates megabyte conversions (`19M` $\to 19.0$), kilobyte fractional conversions (`512k` $\to 0.5$), and null handling on `Varies with device`.
- **`test_clean_playstore_apps`:** Tests corrupted row filtering, duplicate retention (retaining max review counts), and automatic type normalization.
- **`test_clean_user_reviews`:** Validates null review purging, exact duplicate suppression, and sentiment polarity imputation.
- **`test_saved_cleaned_datasets_exist`:** Asserts the physical integrity, row counts (9,659 unique apps, 29,692 reviews), and schema boundaries of the exported datasets on disk.

```bash
$ pytest tests/test_cleaning.py -v
============================= test session starts =============================
tests/test_cleaning.py::test_parse_size_to_mb PASSED                     [ 25%]
tests/test_cleaning.py::test_clean_playstore_apps PASSED                 [ 50%]
tests/test_cleaning.py::test_clean_user_reviews PASSED                   [ 75%]
tests/test_cleaning.py::test_saved_cleaned_datasets_exist PASSED         [100%]
============================== 4 passed in 4.95s ==============================
```
