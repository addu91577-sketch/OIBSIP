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
│   ├── day2_pseudonull_anomaly_audit.md # Day 2 Forensic Pseudonull & Anomaly Audit Report
│   ├── day3_baseline_data_quality_report.md # Day 3 Baseline Data Quality Report & Matrix
│   └── figures/                        # High-resolution diagnostic visualizations
│       ├── day3_column_defect_rates.png
│       ├── day3_record_defect_distribution.png
│       └── day3_defect_cooccurrence_matrix.png
└── src/
    ├── __init__.py
    └── utils.py                        # Reusable data audit & metric routines
```

---

## 📋 Feature Checklist
- [x] **Day 1 (Oct 19):** Workspace setup, raw dataset ingestion, and initial schema inspection
- [x] **Day 2 (Oct 20):** Forensic anomaly detection: catalog pseudonull tokens (`'ERROR'`, `'UNKNOWN'`) and defective value frequencies
- [x] **Day 3 (Oct 21):** Baseline Data Quality Report generation, record defect density & co-occurrence matrix
- [ ] **Day 4 (Oct 22):** Text standardization, whitespace trimming, and pseudonull token conversion to `np.nan`
- [ ] **Day 5 (Oct 23):** Strict data type casting (`datetime64[ns]`, `string` ID, nullable `Int64`, `float64`)
- [ ] **Day 6 (Oct 24):** Categorical encoding (`Item`, `Payment Method`, `Location`) and memory audit
- [ ] **Day 7–10:** Domain-informed and mathematical missing data imputation
- [ ] **Day 11–13:** Multi-tier duplicate resolution and statistical outlier detection
- [ ] **Day 14–16:** Master Before vs. After comparison summary and clean dataset export
- [ ] **Day 17–18:** Automated regression testing (`pytest`), CI workflow, and final delivery certification

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

- **Report:** Detailed findings in `Reports/day2_pseudonull_anomaly_audit.md`.

---

## 📈 Day 3 Milestone: Baseline Data Quality Report & Defect Distribution Matrix

On Day 3, we formalized the enterprise **Baseline Data Quality Report**, evaluated record-level defect penetration, mapped attribute defect co-occurrence, and verified key integrity:

### 1. Master Baseline Data Quality Health Matrix

| Attribute Name | Stored Dtype | True Nulls (`NaN`) | Pseudonulls (`ERROR`/`UNKNOWN`) | Total Corrupted Cells | Defective Rate (%) | Valid Cardinality | Data Health Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Location`** | `object` | 3,265 | 696 | **3,961** | **39.61%** | 2 fulfillment types | Critical missingness; omitted cashier tagging |
| **`Payment Method`** | `object` | 2,579 | 599 | **3,178** | **31.78%** | 3 tender types | High missingness; terminal dropouts |
| **`Item`** | `object` | 333 | 636 | **969** | **9.69%** | 8 menu products | Scanner misses; recoverable via pricing invariant |
| **`Price Per Unit`** | `object` | 179 | 354 | **533** | **5.33%** | 6 menu prices | Deterministic 1-to-1 lookup with `Item` |
| **`Total Spent`** | `object` | 173 | 329 | **502** | **5.02%** | 17 price points | Algebraic identity: $\text{Qty} \times \text{Price}$ |
| **`Quantity`** | `object` | 138 | 341 | **479** | **4.79%** | 5 discrete units | Integer bounds: $1 \le q \le 5$ |
| **`Transaction Date`** | `object` | 159 | 301 | **460** | **4.60%** | 365 trading days | Contiguous 2023 calendar timeline |
| **`Transaction ID`** | `object` | 0 | 0 | **0** | **0.00%** | 10,000 unique keys | **100% Unique Surrogate Primary Key** |
| **TOTALS** | — | **6,826** | **2,897** | **9,723** | **12.15%** | — | **Baseline Certified** |

### 2. Record-Level Defect Density & Naive Deletion Risk

| Defect Count Per Record | Number of Transactions | Share of Dataset (%) | Cumulative Share (%) | Pipeline Implication |
| :---: | :---: | :---: | :---: | :--- |
| **0 (Pristine)** | **3,089** | **30.89%** | 30.89% | Valid ground-truth calibration baseline |
| **1 Defect** | **4,285** | **42.85%** | 73.74% | Deterministic single-attribute recovery |
| **2 Defects** | **2,136** | **21.36%** | 95.10% | Sequential invariant & formula imputation |
| **3 Defects** | **439** | **4.39%** | 99.49% | Multi-step algebraic derivation & mode fill |
| **4 Defects** | **47** | **0.47%** | 99.96% | Boundary constraint propagation |
| **5 Defects** | **4** | **0.04%** | 100.00% | Composite mode & temporal interpolation |

> [!CAUTION]
> **Why `df.dropna()` is Fatal:** Running standard row deletion would discard **6,911 transactions (69.11% of the entire dataset)**, biasing business intelligence metrics and discarding over **$61,700 USD** in valid revenue. Row deletion is strictly eliminated in favor of lossless imputation.

### 3. Key Day 3 Audit Achievements:
- **Zero Full-Row Duplicates:** Confirmed **0 redundant rows** across 10,000 transactions.
- **100.0% Key Integrity:** Verified **10,000 unique `Transaction ID` surrogate keys** (0 collisions).
- **Defect Co-Occurrence Heatmap:** Uncovered major clustering between `Location` and `Payment Method` (1,294 co-occurring defects), while confirming extreme sparsity of joint pricing/volume failures ($\le 54$ records).
- **Diagnostic Visual Suite:** Programmatically synthesized and serialized 3 publication-ready figures to `Reports/figures/`.
- **Full Report:** Comprehensive analysis available in [`Reports/day3_baseline_data_quality_report.md`](Reports/day3_baseline_data_quality_report.md).

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
