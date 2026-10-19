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
- [x] **Load dataset and perform initial inspection:** shape, column dtypes, null value check
- [ ] **Descriptive statistics:** mean, median, mode, standard deviation for all numerical columns
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
└── Notebooks/
    └── 01_retail_sales_eda.ipynb       # Jupyter Notebook exploratory data analysis workflow
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

## 🔍 Initial Inspection & Data Hygiene Findings
* **Dataset Shape:** Exactly **1,000 rows** and **9 columns**.
* **Zero Missing Values:** Confirmed 0 null or NaN entries across all 9 features.
* **Integrity & Uniqueness:** 0 duplicate rows. `Transaction ID` serves as a strictly unique integer key from 1 to 1,000.
* **Customer Demographics:** Ages span from **18 to 64 years** (average age: ~41.4 years) with balanced gender representation.
* **Catalog Coverage:** Core retail sales span three categories: **Clothing**, **Electronics**, and **Beauty**.
* **Financial Scale:** Order spend ranges from **$25 to $2,000** with an average basket value of **$456.00** (`Total Amount = Quantity × Price per Unit`).

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
