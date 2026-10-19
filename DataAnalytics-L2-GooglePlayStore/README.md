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
- [ ] **Data cleaning:** fix incorrect data types (`Installs` as string with `+`, `Price` with `$`, `Reviews` as string), handle nulls, remove duplicates
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
├── .gitignore                          # Git ignore rules for Python, virtual environments, and Jupyter
├── README.md                           # Project documentation, inspection findings, and architecture
├── requirements.txt                    # Project environment dependencies
├── Data/
│   ├── googleplaystore.csv             # Raw Google Play Store apps catalog (10,841 rows × 13 columns)
│   ├── googleplaystore_user_reviews.csv# Raw user sentiment reviews dataset (64,295 rows × 5 columns)
│   └── license.txt                     # Dataset citation & license terms
└── Notebooks/
    └── 01_playstore_inspection_ingestion.ipynb # Day 1: Raw dataset ingestion & structural schema audit
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
