# 🛍️ Retail Sales Data — Exploratory Data Analysis (EDA)
**Oasis Infobyte Data Analytics Internship — Task Level 1**

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-2.x-150458.svg)
![NumPy](https://img.shields.io/badge/numpy-1.x-013243.svg)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.x-11557c.svg)
![Seaborn](https://img.shields.io/badge/seaborn-0.12+-4c72b0.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)

---

## 🎯 Objective
Perform a rigorous Exploratory Data Analysis (EDA) on retail sales data to uncover purchasing patterns, customer demographic trends, product revenue concentration, and actionable business insights. Every analytical step is documented with clean Python code, statistical assessments, and narrative markdown observations.

---

## 🛠️ Tech Stack
- **Language:** Python 3
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Interactive Environment:** Jupyter Notebook (`.ipynb`)
- **Version Control:** Git & GitHub

---

## 📋 Feature Checklist
- [x] **Load dataset and perform initial inspection:** shape, column dtypes, null value check (Day 1)
- [x] **Descriptive statistics:** mean, median, mode, standard deviation for all numerical columns (Day 2)
- [x] **Time series analysis:** plot monthly and quarterly sales trends using line charts (Day 3)
- [ ] **Customer demographics analysis:** distribution of customer age groups, gender breakdown
- [ ] **Product analysis:** top 10 best-selling products; revenue by product category (bar chart)
- [ ] **Heatmap:** correlation matrix between numerical variables
- [ ] **At least one additional visualisation:** reveals a non-obvious insight
- [ ] **Markdown cells throughout notebook:** written observations after each chart
- [ ] **Conclusion section:** at least 3 specific, actionable business recommendations based on findings

---

## 📂 Repository Structure
```text
DataAnalytics-L1-EDARetailSales/
├── .gitignore                          # Git ignore rules for Python, Jupyter, and temporary files
├── README.md                           # Project documentation and inspection highlights
├── requirements.txt                    # Project dependencies
├── Data/
│   ├── retail_sales_dataset.csv        # Primary retail dataset (1,000 transactions × 9 attributes)
│   └── Sample - Superstore.csv         # Benchmark / reference sales dataset
├── Notebooks/
│   └── 01_retail_sales_eda.ipynb       # Jupyter Notebook exploratory data analysis workflow
└── screenshots/                        # Exported statistical distribution & time-series visualizations
    ├── 01_numerical_distributions_hist_kde.png
    ├── 02_numerical_boxplots.png
    ├── 03_monthly_sales_trend.png
    ├── 04_quarterly_sales_trend.png
    ├── 05_monthly_category_sales_trend.png
    └── 06_day_of_week_sales_velocity.png
```

---

## 📊 Dataset Overview
The primary dataset captures transactional point-of-sale retail records across major consumer goods categories.

| Column Name | Data Type | Description | Example Values |
| :--- | :--- | :--- | :--- |
| **Transaction ID** | `int64` | Unique transaction identifier | `1`, `2`, `3` |
| **Date** | `object` | Purchase date timestamp | `2023-11-24`, `2023-02-27` |
| **Customer ID** | `object` | Customer alphanumeric identifier | `CUST001`, `CUST002` |
| **Gender** | `object` | Customer gender identity | `Male`, `Female` |
| **Age** | `int64` | Customer age in years | `34`, `26`, `50` |
| **Product Category**| `object` | Retail merchandise department | `Clothing`, `Electronics`, `Beauty` |
| **Quantity** | `int64` | Number of units purchased | `1`, `2`, `3`, `4` |
| **Price per Unit** | `int64` | Selling price per item ($) | `25`, `30`, `50`, `300`, `500` |
| **Total Amount** | `int64` | Total transaction value ($) | `150`, `1000`, `30`, `500` |

---

## 🔍 Initial Inspection & Data Hygiene Findings (Day 1)
* **Dataset Shape:** Exactly **1,000 rows** and **9 columns**.
* **Zero Missing Values:** Confirmed 0 null or NaN entries across all 9 features.
* **Integrity & Uniqueness:** 0 duplicate rows. `Transaction ID` serves as a strictly unique integer key from 1 to 1,000.
* **Customer Demographics:** Ages span from **18 to 64 years** (average age: ~41.4 years) with balanced gender representation.
* **Catalog Coverage:** Core retail sales span three categories: **Clothing**, **Electronics**, and **Beauty**.
* **Financial Scale:** Order spend ranges from **$25 to $2,000** with an average basket value of **$456.00** (`Total Amount = Quantity × Price per Unit`).

---

## 📊 Day 2: Descriptive Statistics & Numerical Profiling Findings

### 📐 Comprehensive Statistical Summary Table
| Feature | Count | Mean | Median | Mode | Std Dev | Variance | Min | 25% (Q1) | 75% (Q3) | IQR | Max | Skewness | Kurtosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Age`** | 1,000 | 41.39 | 42.00 | 43 | 13.68 | 187.18 | 18 | 29.00 | 53.00 | 24.00 | 64 | -0.049 | -1.201 |
| **`Quantity`** | 1,000 | 2.51 | 3.00 | 4 | 1.13 | 1.28 | 1 | 1.00 | 4.00 | 3.00 | 4 | -0.014 | -1.393 |
| **`Price per Unit`** | 1,000 | $179.89 | $50.00 | $50 | $189.68 | 35,979.02 | $25 | $30.00 | $300.00 | $270.00 | $500 | +0.736 | -1.139 |
| **`Total Amount`** | 1,000 | $456.00 | $135.00 | $50 | $560.00 | 313,597.35 | $25 | $60.00 | $900.00 | $840.00 | $2,000 | +1.376 | +0.815 |

### 📈 Key Day 2 Takeaways & Commercial Insights:
* **Symmetry in Demographics & Basket Volumes:**
  - Customer age (`Skewness: -0.049`) and basket quantities (`Skewness: -0.014`) display near-perfect balanced distributions across all demographic cohorts and order sizes ($\text{Mean} \approx \text{Median} \approx \text{Mode}$).
  - Shopper ages span evenly from 18 to 64 years across Gen Z, Millennials, Gen X, and Boomers without demographic clustering.
* **Positive Revenue Skew:**
  - Transaction revenue (`Total Amount`) is substantially right-skewed (`Skewness: +1.376`).
  - While 50% of orders are under **$135.00** (median), the mean spend is driven up to **$456.00** by premium basket combinations scaling up to **$2,000.00**.
* **Tiered Catalog Pricing Architecture:**
  - Catalog pricing is discrete across 5 tiers: **$25, $30, $50, $300, and $500**.
  - Everyday budget essentials ($25–$50) account for **60.4%** of volume, whereas high-margin premium goods ($300–$500) drive the substantial financial upside.
* **Zero Outlier Anomalies:**
  - Applied Tukey's $1.5 \times \text{IQR}$ rule across all variables; exactly 0 records exceed legitimate data boundaries.

### 🖼️ Exported Visualizations:
* [01_numerical_distributions_hist_kde.png](screenshots/01_numerical_distributions_hist_kde.png) — Histograms with Kernel Density Estimation (KDE) and superimposed Mean, Median, and Mode markers.
* [02_numerical_boxplots.png](screenshots/02_numerical_boxplots.png) — Boxplots profiling Interquartile Range (IQR), quartile spreads, and outlier boundary checks.

---

## 📈 Day 3: Time Series & Sales Trends Analysis Findings

### 📅 Monthly Sales Velocity & Performance Metrics
| Month | Total Revenue | Orders | Units Sold | Avg Order Value (AOV) | MoM Revenue Growth | MoM Volume Growth |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2023-01`** | $35,450.00 | 76 | 195 | $466.45 | — | — |
| **`2023-02`** | $44,060.00 | 85 | 228 | $518.35 | +24.29% | +11.84% |
| **`2023-03`** | $28,990.00 | 73 | 180 | $397.12 | -34.20% | -14.12% |
| **`2023-04`** | $33,870.00 | 86 | 213 | $393.84 | +16.83% | +17.81% |
| **`2023-05`** 🏆 | **$53,150.00** | **105** | **276** | **$506.19** | **+56.92%** | **+22.09%** |
| **`2023-06`** | $36,715.00 | 77 | 181 | $476.82 | -30.92% | -26.67% |
| **`2023-07`** | $35,465.00 | 72 | 183 | $492.57 | -3.40% | -6.49% |
| **`2023-08`** | $36,960.00 | 94 | 209 | $393.19 | +4.22% | +30.56% |
| **`2023-09`** ⚠️ | **$23,620.00** | **65** | **181** | **$363.38** | **-36.09%** | **-30.85%** |
| **`2023-10`** 🚀 | $46,580.00 | 96 | 231 | $485.21 | **+97.21%** | **+47.69%** |
| **`2023-11`** | $34,920.00 | 78 | 202 | $447.69 | -25.03% | -18.75% |
| **`2023-12`** 🎄 | $44,690.00 | 91 | 231 | $491.10 | +27.98% | +16.67% |
| **`2024-01`** *(Cutoff)* | $1,530.00 | 2 | 4 | $765.00 | -96.58% | -97.80% |

---

### 📊 Quarterly Sales Trajectory
| Quarter | Total Revenue | Orders | Units Sold | Avg Order Value (AOV) | QoQ Revenue Growth | Retail Cycle Dynamics |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`2023 Q1`** | $108,500.00 | 234 | 603 | $463.68 | — | **Stable Baseline:** Strong winter baseline anchored by February post-holiday promotions. |
| **`2023 Q2`** | $123,735.00 | **268** | **670** | $461.70 | **+14.04%** | **Volume Zenith:** Highest transaction count (268 orders) driven by massive May shopping momentum. |
| **`2023 Q3`** | $96,045.00 | 231 | 573 | $415.78 | **-22.38%** | **Summer Lull:** Annual trough across both revenue and basket spend, dragged by September contraction. |
| **`2023 Q4`** 👑 | **$126,190.00** | 265 | 664 | **$476.19** | **+31.39%** | **Revenue Champion:** Highest quarterly revenue, bolstered by Halloween, Black Friday, and Christmas holidays. |
| **`2024 Q1`** *(Partial)* | $1,530.00 | 2 | 4 | $765.00 | -98.79% | **Dataset Horizon Boundary:** Contains only 1 calendar day (2024-01-01); isolated from quarterly benchmarks. |

---

### 📈 Key Day 3 Takeaways & Commercial Insights:
* **Bimodal Seasonality (Spring & Q4 Surges):** Retail revenue peaks twice annually: first in **May 2023 ($53,150.00)** driven by spring and holiday promotions, and subsequently across **Q4 2023 ($126,190.00)** driven by year-end holiday spending.
* **Volume vs. Ticket Interplay:** Peak sales months occur when high transaction volumes (>85 orders) intersect with above-average basket sizes (>$480 AOV). In contrast, the annual trough in **September 2023 ($23,620.00)** was driven by a concurrent contraction in order traffic (65 orders) and ticket size ($363.38).
* **Category Seasonality & Hedging Dynamics:**
  * **`Electronics` (High Volatility):** Drives store revenue peaks ($23.2k in May, $20.2k in Dec) but experiences severe contractions ($3.4k in March, $7.3k in September).
  * **`Clothing` (Reliable Baseline):** Sustained demand throughout the year ($8.2k–$17.5k monthly), peaking in May ($17.5k) and finishing strong in Q4 ($27.3k total).
  * **`Beauty` (Counter-Cyclical Summer Stabilizer):** Peaked in July ($16,090) precisely when Clothing reached its annual low, serving as an organic portfolio hedge.
* **Saturday Shopping Dominance:** Saturdays generate the highest revenue (**$78,815.00**) and highest basket spend (**$525.43 AOV**), whereas Thursday represents the weekly commercial lull (**$53,835.00**; 123 orders).
* **Data Horizon Boundary:** The dataset terminates on January 1, 2024 (2 records, $1,530), which is isolated as a boundary artifact to prevent skewing annual/quarterly comparability.

### 🖼️ Exported Visualizations:
* [03_monthly_sales_trend.png](screenshots/03_monthly_sales_trend.png) — Monthly revenue trajectory with 2023 baseline, peak/trough annotations, and dual-axis transaction volume vs. AOV.
* [04_quarterly_sales_trend.png](screenshots/04_quarterly_sales_trend.png) — Quarterly sales performance and transaction count progression.
* [05_monthly_category_sales_trend.png](screenshots/05_monthly_category_sales_trend.png) — Product category monthly revenue trajectories showing seasonal divergence and category hedging.
* [06_day_of_week_sales_velocity.png](screenshots/06_day_of_week_sales_velocity.png) — Day-of-week revenue turnover and Average Order Value showing Saturday apex.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Environment Setup
```bash
cd DataAnalytics-L1-EDARetailSales
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Notebook
```bash
jupyter notebook Notebooks/01_retail_sales_eda.ipynb
```
