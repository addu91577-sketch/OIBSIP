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
- [ ] **Time series analysis:** plot monthly and quarterly sales trends using line charts
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
└── screenshots/                        # Exported statistical distribution visualizations
    ├── 01_numerical_distributions_hist_kde.png
    └── 02_numerical_boxplots.png
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
