# ☕ Cafe Sales – Data Cleaning & Transformation Pipeline
**Oasis Infobyte Data Analytics Internship · Task Level 1**

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-2.x-150458.svg)
![NumPy](https://img.shields.io/badge/numpy-1.x-013243.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)

---

## 🎯 Objective
Demonstrate professional-level data cleaning skills by taking a deliberately messy transaction dataset (**Cafe Sales – Dirty Data**) and systematically transforming it into a clean, analysis-ready dataset. Every transformation, imputation strategy, duplicate resolution, outlier decision, and standardization step is rigorously documented.

---

## 🛠️ Tech Stack
- **Language:** Python 3.14
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Interactive Environment:** Jupyter Notebook (`.ipynb`)
- **Version Control:** Git & GitHub

---

## 📁 Repository Structure
```text
DataAnalytics-L1-CleaningData/
├── .gitignore                          # Git ignore rules for Python, Jupyter, and temporary files
├── README.md                           # Master project documentation
├── requirements.txt                    # Project dependencies
├── Data/
│   └── dirty_cafe_sales.csv            # Raw dataset (10,000 rows × 8 columns)
├── Notebooks/
│   └── 01_cafe_sales_data_cleaning.ipynb # Jupyter Notebook data cleaning workflow
├── Reports/
│   └── day2_pseudonull_anomaly_audit.md # Day 2 Forensic Pseudonull & Anomaly Audit Report
└── src/
    ├── __init__.py
    └── utils.py                        # Reusable data audit & metric routines
```

---

## 📊 Dataset Overview
The raw dataset simulates realistic point-of-sale (POS) cafe transaction records containing deliberate data entry anomalies, missing values, and pseudonulls across 10,000 transactions and 8 attributes.

| Column Name | Ingested Dtype | Description | Example Raw Values |
| :--- | :---: | :--- | :--- |
| **`Transaction ID`** | `object` | Unique transaction primary key | `TXN_0000001`, `TXN_0000002` |
| **`Item`** | `object` | Menu item purchased | `Coffee`, `Tea`, `Sandwich`, `ERROR`, `UNKNOWN` |
| **`Quantity`** | `object` | Units purchased per item | `1`, `2`, `5`, `NaN`, `ERROR` |
| **`Price Per Unit`** | `object` | Unit price in USD | `2.0`, `1.5`, `NaN`, `UNKNOWN` |
| **`Total Spent`** | `object` | Net transaction cost ($) | `4.0`, `15.0`, `NaN`, `ERROR` |
| **`Payment Method`** | `object` | Mode of customer settlement | `Cash`, `Credit Card`, `Digital Wallet` |
| **`Location`** | `object` | Order fulfillment mode | `In-store`, `Takeaway` |
| **`Transaction Date`** | `object` | Transaction timestamp/date | `2023-01-15`, `UNKNOWN` |

---

## 🔍 Day 2 Milestone: Forensic Pseudonull & Anomaly Catalog

On Day 2, a forensic anomaly audit cataloged true missing values (`NaN`) alongside hidden string sentinel tokens (`'ERROR'`, `'UNKNOWN'`) across the dataset:

| Column Name | Ingested Dtype | True Nulls (`NaN`) | `'ERROR'` Tokens | `'UNKNOWN'` Tokens | Total Dirty Records | Defective Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Location`** | `object` | 3,265 | 358 | 338 | **3,961** | **39.61%** |
| **`Payment Method`** | `object` | 2,579 | 306 | 293 | **3,178** | **31.78%** |
| **`Item`** | `object` | 333 | 292 | 344 | **969** | **9.69%** |
| **`Price Per Unit`** | `object` | 179 | 190 | 164 | **533** | **5.33%** |
| **`Total Spent`** | `object` | 173 | 164 | 165 | **502** | **5.02%** |
| **`Quantity`** | `object` | 138 | 170 | 171 | **479** | **4.79%** |
| **`Transaction Date`** | `object` | 159 | 142 | 159 | **460** | **4.60%** |
| **`Transaction ID`** | `object` | 0 | 0 | 0 | **0** | **0.00%** |
| **TOTALS** | — | **6,826** | **1,622** | **1,275** | **9,723** | **12.15%** |

### Key Day 2 Audit Insights:
- **2,897 Pseudonulls Uncovered:** 1,622 `'ERROR'` and 1,275 `'UNKNOWN'` string tokens masqueraded as valid string data, escaping standard `df.isnull()` checks.
- **True Defect Baseline:** Real defect volume is **9,723 corrupted cells** (12.15% overall corruption rate) rather than 6,826 nulls.
- **100% Primary Key Integrity:** `Transaction ID` has 0 nulls, 0 errors, and 10,000 unique keys (0 collisions).
- **Audit Report:** Comprehensive documentation available in `Reports/day2_pseudonull_anomaly_audit.md`.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python 3.10+ is installed.

### 2. Environment Setup
Clone the repository and install the dependencies:
```bash
cd DataAnalytics-L1-CleaningData
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch Notebook
```bash
jupyter notebook Notebooks/01_cafe_sales_data_cleaning.ipynb
```
