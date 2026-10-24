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
│   ├── day4_text_standardization_and_token_normalization.md # Day 4 Text Standardization & Token Normalization Report
│   ├── day5_strict_datatype_enforcement.md # Day 5 Strict Data Type Enforcement & Validation Report
│   ├── day6_categorical_encoding_and_memory_audit.md # Day 6 Categorical Encoding & Memory Audit Report
│   └── figures/                        # High-resolution diagnostic visualizations
│       ├── day3_column_defect_rates.png
│       ├── day3_record_defect_distribution.png
│       ├── day3_defect_cooccurrence_matrix.png
│       ├── day4_pseudonull_normalization_waterfall.png
│       ├── day4_standardized_categorical_profiles.png
│       ├── day5_datatype_transformation_matrix.png
│       ├── day5_numerical_and_temporal_profiles.png
│       ├── day6_categorical_profiles_and_cardinality.png
│       └── day6_memory_footprint_reduction_waterfall.png
└── src/
    ├── __init__.py
    └── utils.py                        # Reusable data audit & metric routines
```

---

## 📋 Feature Checklist
- [x] **Day 1 (Oct 19):** Workspace setup, raw dataset ingestion, and initial schema inspection
- [x] **Day 2 (Oct 20):** Forensic anomaly detection: catalog pseudonull tokens (`'ERROR'`, `'UNKNOWN'`) and defective value frequencies
- [x] **Day 3 (Oct 21):** Baseline Data Quality Report generation, record defect density & co-occurrence matrix
- [x] **Day 4 (Oct 22):** Text standardization, whitespace trimming, and pseudonull token conversion to `np.nan`
- [x] **Day 5 (Oct 23):** Strict data type casting (`datetime64[ns]`, `string` ID, nullable `Int64`, `float64`)
- [x] **Day 6 (Oct 24):** Categorical encoding (`Item`, `Payment Method`, `Location`) and memory audit
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

## 🧹 Day 4 Milestone: Text Standardization & Pseudonull Token Normalization

On Day 4, the active cleaning workflow began with non-destructive text sanitization, whitespace trimming, and mapping disguised sentinel placeholders (`'ERROR'`, `'UNKNOWN'`) to standard IEEE `np.nan`:

### Master Pseudonull Normalization & Replacement Matrix

| Attribute Name | Raw Ingested Dtype | Raw True NaNs | Converted `'ERROR'` Tokens | Converted `'UNKNOWN'` Tokens | Total Sentinels Converted | Post-Day 4 Nulls (`np.nan`) | Post-Day 4 Null Rate (%) | Transformation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Location`** | `object` | 3,265 | 358 | 338 | **696** | **3,961** | **39.61%** | Normalized to `np.nan` |
| **`Payment Method`** | `object` | 2,579 | 306 | 293 | **599** | **3,178** | **31.78%** | Normalized to `np.nan` |
| **`Item`** | `object` | 333 | 292 | 344 | **636** | **969** | **9.69%** | Normalized to `np.nan` |
| **`Price Per Unit`** | `object` | 179 | 190 | 164 | **354** | **533** | **5.33%** | Normalized to `np.nan` |
| **`Total Spent`** | `object` | 173 | 164 | 165 | **329** | **502** | **5.02%** | Normalized to `np.nan` |
| **`Quantity`** | `object` | 138 | 170 | 171 | **341** | **479** | **4.79%** | Normalized to `np.nan` |
| **`Transaction Date`** | `object` | 159 | 142 | 159 | **301** | **460** | **4.60%** | Normalized to `np.nan` |
| **`Transaction ID`** | `object` | 0 | 0 | 0 | **0** | **0** | **0.00%** | **100% Unique Primary Key Intact** |
| **TOTALS** | — | **6,826** | **1,622** | **1,634** | **3,256** | **10,082** | **12.60%** | **Consolidated Missingness Baseline** |

### Key Day 4 Achievements:
- **3,256 Sentinel Error Tokens Converted:** Exactly 1,622 `'ERROR'` and 1,634 `'UNKNOWN'` placeholders successfully mapped to IEEE `np.nan`.
- **100% Whitespace Sanitization:** Audited and stripped leading/trailing whitespace across all text columns without data loss.
- **Categorical Casing Standardized:** Enforced canonical casing on `Item` (8 menu products), `Payment Method` (3 tender types), and `Location` (`Takeaway`, `In-store`).
- **Zero Token Leakage Verified:** Programmatic regression suite confirmed exactly 0 residual sentinels across the entire dataset.
- **Diagnostic Visual Artifacts:** Serialized `day4_pseudonull_normalization_waterfall.png` and `day4_standardized_categorical_profiles.png` to `Reports/figures/`.
- **Full Report:** Comprehensive analysis available in [`Reports/day4_text_standardization_and_token_normalization.md`](Reports/day4_text_standardization_and_token_normalization.md).

---

## 🛡️ Day 5 Milestone: Strict Data Type Enforcement & Validation

On Day 5, the pipeline transitioned from unstructured `object` arrays to strictly typed primitives, unlocking vectorized performance, memory efficiency, and schema integrity:

| Attribute Name | Ingested Dtype | Pre-Cast Dtype | Operational Enforced Dtype | Missing Cells | Null Representation | Boundary / Domain Constraint | Validation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`Transaction ID`** | `object` | `object` (`str`) | **`string`** | 0 (0.0%) | None | Primary Key (`^TXN_\d{7}$`) | **PASSED (10,000 Unique Keys)** |
| **`Item`** | `object` | `object` (`str`) | **`object` (`str`)** | 969 (9.7%) | `np.nan` | Standardized String | **READY (Day 6 Category)** |
| **`Quantity`** | `object` | `object` (`str`) | **`Int64`** | 479 (4.8%) | `<NA>` | Discrete integer $1 \dots 5$ | **PASSED (0 Range Violations)** |
| **`Price Per Unit`** | `object` | `object` (`str`) | **`float64`** | 533 (5.3%) | `np.nan` | Canonical Menu {$\$1.00..\$5.00$} | **PASSED (100% Menu Aligned)** |
| **`Total Spent`** | `object` | `object` (`str`) | **`float64`** | 502 (5.0%) | `np.nan` | Monetary scale $[\$1.00..\$25.00]$ | **PASSED (0 Violations)** |
| **`Payment Method`** | `object` | `object` (`str`) | **`object` (`str`)** | 3,178 (31.8%) | `np.nan` | Standardized String | **READY (Day 6 Category)** |
| **`Location`** | `object` | `object` (`str`) | **`object` (`str`)** | 3,961 (39.6%) | `np.nan` | Standardized String | **READY (Day 6 Category)** |
| **`Transaction Date`** | `object` | `object` (`str`) | **`datetime64[ns]`**| 460 (4.6%) | `NaT` | Contiguous 2023 (365 Days) | **PASSED (365 Trading Days)** |

### Key Day 5 Achievements:
- **Zero Coercion Loss ($\Delta = 0$):** Exactly 0 valid data values were lost, truncated, or unparseable during type conversion.
- **Nullable Integer Semantics:** Implemented pandas `Int64` for `Quantity`, natively preserving discrete counts ($1 \dots 5$) without float coercion.
- **Temporal Index Readiness:** Converted `Transaction Date` to `datetime64[ns]`, confirming contiguous customer purchases spanning all **365 trading days in 2023** (`2023-01-01` to `2023-12-31`) averaging 26.1 transactions/day.
- **Immediate Memory Optimization:** Reduced DataFrame memory allocation from **1,064.0 KB to 907.3 KB (-14.73%)** prior to Day 6 categorical encoding.
- **100% Quality Assertion Sign-Off:** Automated verification suite (`verify_type_integrity()`) certified zero schema violations, 100% primary key integrity, and exact domain boundaries.
- **Diagnostic Visual Artifacts:** Serialized `day5_datatype_transformation_matrix.png` and `day5_numerical_and_temporal_profiles.png` to `Reports/figures/`.
- **Full Report:** Comprehensive analysis available in [`Reports/day5_strict_datatype_enforcement.md`](Reports/day5_strict_datatype_enforcement.md).

---

## 🏷️ Day 6 Milestone: Categorical Encoding & Comprehensive Memory Audit

On Day 6, the pipeline enforced pandas `CategoricalDtype` across `Item`, `Payment Method`, and `Location` with explicit canonical category universes, transitioning storage from heap pointers to compact `int8` dictionary-encoded representations:

| Feature Name | Pre Dtype | Enforced Dtype | Underlying Code Type | Canonical Classes | Non-Null Records | Missing Records (`NaN`) | Modal Category | Pre Memory (KB) | Post Memory (KB) | Memory Reduction (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Item`** | `str` | `category` | `int8` | **8** | 9,031 | 969 (9.69%) | `Juice` (1,171) | 132.84 | **9.87** | **-92.25%** |
| **`Payment Method`** | `str` | `category` | `int8` | **3** | 6,822 | 3,178 (31.78%) | `Digital Wallet` (2,291) | 147.53 | **9.82** | **-93.19%** |
| **`Location`** | `str` | `category` | `int8` | **2** | 6,039 | 3,961 (39.61%) | `Takeaway` (3,022) | 130.71 | **9.80** | **-92.33%** |
| **TOTALS** | — | — | `int8` | **13** | **21,892** | **8,108** | — | **411.08** | **29.49** | **-92.83%** |

### Multi-Stage Deep Memory Compression Scorecard:
- **Raw Ingestion (Day 1):** 1,083.09 KB
- **Sanitized Text (Day 4):** 1,083.09 KB
- **Strict Numerical & Date Primitives (Day 5):** 907.30 KB (-16.23%)
- **Categorical Encoded (Day 6):** **537.93 KB (-50.33% cumulative reduction)**

### Key Day 6 Achievements:
- **Canonical Domain Closed:** Strictly bound `Item` (8 products: Cake to Tea), `Payment Method` (3 methods: Cash, Credit Card, Digital Wallet), and `Location` (2 modes: In-store, Takeaway).
- **Zero Information Loss:** Preserved all 8,108 missing entries as native `NaN` (underlying Sentinel `-1`). Exactly 0 invalid category levels created and 0 valid strings corrupted ($\Delta = 0$).
- **Compact Integer Codes:** Replaced heap string objects with SIMD-aligned 1-byte `int8` category codes.
- **-92.83% Categorical Memory Slashed:** Slashed combined categorical footprint from 411.08 KB down to 29.49 KB.
- **-50.33% DataFrame Memory Reduction:** Overall memory reduced by more than half, enabling high-performance vectorized operations.
- **100% Assertion Sign-Off:** Automated verification suite (`verify_categorical_integrity()`) certified zero domain or type violations.
- **Diagnostic Visual Artifacts:** Serialized `day6_categorical_profiles_and_cardinality.png` and `day6_memory_footprint_reduction_waterfall.png` to `Reports/figures/`.
- **Full Report:** Comprehensive analysis available in [`Reports/day6_categorical_encoding_and_memory_audit.md`](Reports/day6_categorical_encoding_and_memory_audit.md).

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
