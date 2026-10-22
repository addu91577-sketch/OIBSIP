# 🧹 Cafe Sales – Day 4 Report: Text Standardization & Pseudonull Token Normalization
**Oasis Infobyte Data Analytics Internship · Task Level 1**  
**Milestone:** Day 4 of 18 — Text Standardization, Whitespace Trimming, and Pseudonull Conversion to `np.nan`  
**Target Asset:** `Data/dirty_cafe_sales.csv` (10,000 rows × 8 columns)  
**Status:** Certified Day 4 Transformations Complete & Verified  

---

## 1. Executive Summary & Audit Scope

On **Day 4**, our data cleaning engineering workflow progressed from forensic audit cataloging to the first active structural transformations of the dataset.

In legacy Point-of-Sale (POS) systems and fragmented databases, data corruption commonly manifests as literal text placeholders (e.g. `'ERROR'`, `'UNKNOWN'`) and inconsistent text spacing. In the raw dataset, these pseudonulls contaminated seven out of eight attributes, causing pandas and analytical tools to treat genuine missing values as valid strings. This prevented standard numerical computations, datetime conversions, and statistical aggregations.

### Core Day 4 Achievements:
1. **Systematic Whitespace Trimming:** Sanitized 100% of text and string entries across all 10,000 records, eliminating trailing and leading whitespace anomalies.
2. **Casing & Categorical Normalization:** Enforced strict canonical casing across categorical dimensions (`Item`, `Payment Method`, `Location`) and surrogate primary keys (`Transaction ID`).
3. **Pseudonull Normalization (3,256 Tokens Converted):** Successfully converted **3,256 literal sentinel error tokens** (1,622 `'ERROR'` + 1,634 `'UNKNOWN'`) to canonical IEEE `np.nan` values across the seven defective attributes.
4. **Consolidated Missingness Baseline (10,082 Null Cells):** Consolidated all missing data into standard IEEE `np.nan`, establishing the true missingness baseline (**10,082 null cells / 12.60% of the 80,000-cell grid**) without modifying pristine customer transactions.
5. **Zero Token Leakage Verified:** Executed automated programmatic regression assertions certifying that exactly **0 sentinel tokens remain** in the dataset and that entity integrity is 100% intact.

---

## 2. Text Whitespace Sanitization & Casing Normalization

Leading and trailing whitespace introduces subtle, high-impact defects in data pipelines, causing identical categorical values (e.g. `'Coffee '` vs `'Coffee'`) to be treated as distinct classes, skewing grouping, filtering, and joins.

On Day 4, all string and object columns were processed with a non-destructive whitespace trimming pass that strips whitespace from text entries while strictly preserving null states.

### Whitespace Audit Matrix:

| Column Name | Ingested Dtype | Audited Records | Leading Whitespace Anomalies | Trailing Whitespace Anomalies | Post-Day 4 Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Transaction ID`** | `object` (`str`) | 10,000 | 0 | 0 | **Clean (Verified Pattern `^TXN_\d{7}$`)** |
| **`Item`** | `object` (`str`) | 10,000 | 0 | 0 | **Clean (Canonical Title Casing Enforced)** |
| **`Quantity`** | `object` (`str`) | 10,000 | 0 | 0 | **Clean (Pre-Cast Integer Strings)** |
| **`Price Per Unit`** | `object` (`str`) | 10,000 | 0 | 0 | **Clean (Pre-Cast Currency Strings)** |
| **`Total Spent`** | `object` (`str`) | 10,000 | 0 | 0 | **Clean (Pre-Cast Currency Strings)** |
| **`Payment Method`** | `object` (`str`) | 10,000 | 0 | 0 | **Clean (Canonical Title Casing Enforced)** |
| **`Location`** | `object` (`str`) | 10,000 | 0 | 0 | **Clean (Canonical Form: `Takeaway`, `In-store`)** |
| **`Transaction Date`** | `object` (`str`) | 10,000 | 0 | 0 | **Clean (Pre-Cast ISO-8601 Strings)** |

---

## 3. Master Pseudonull Normalization Matrix (Before vs. After)

Prior to Day 4, the raw dataset contained a mix of true IEEE `NaN` values and literal string placeholders. The table below documents the transformation and exact counts of converted sentinel tokens:

| Attribute Name | Raw Ingested Dtype | Raw True NaNs | Converted `'ERROR'` Tokens | Converted `'UNKNOWN'` Tokens | Total Sentinels Converted | Post-Day 4 Nulls (`np.nan`) | Post-Day 4 Null Rate (%) | Transformation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Location`** | `object` | 3,265 | 358 | 338 | **696** | **3,961** | **39.61%** | Successfully Normalized to `np.nan` |
| **`Payment Method`** | `object` | 2,579 | 306 | 293 | **599** | **3,178** | **31.78%** | Successfully Normalized to `np.nan` |
| **`Item`** | `object` | 333 | 292 | 344 | **636** | **969** | **9.69%** | Successfully Normalized to `np.nan` |
| **`Price Per Unit`** | `object` | 179 | 190 | 164 | **354** | **533** | **5.33%** | Successfully Normalized to `np.nan` |
| **`Total Spent`** | `object` | 173 | 164 | 165 | **329** | **502** | **5.02%** | Successfully Normalized to `np.nan` |
| **`Quantity`** | `object` | 138 | 170 | 171 | **341** | **479** | **4.79%** | Successfully Normalized to `np.nan` |
| **`Transaction Date`** | `object` | 159 | 142 | 159 | **301** | **460** | **4.60%** | Successfully Normalized to `np.nan` |
| **`Transaction ID`** | `object` | 0 | 0 | 0 | **0** | **0** | **0.00%** | **100% Unique Primary Key Intact** |
| **TOTALS** | — | **6,826** | **1,622** | **1,634** | **3,256** | **10,082** | **12.60%** | **Consolidated Missingness Baseline** |

### Mathematical Validation of Normalization:
$$\text{Post-Day 4 Nulls}_j = \text{Raw NaNs}_j + \text{ERROR}_j + \text{UNKNOWN}_j$$
$$\sum_{j=1}^{8} \text{Post-Day 4 Nulls}_j = 6,826 + 3,256 = 10,082 \text{ cells}$$

This confirms that no non-sentinel data values were inadvertently overwritten and that all 3,256 pseudonulls were mapped to `np.nan`.

---

## 4. Post-Normalization Clean Categorical Domain Validation

With sentinel noise eliminated, we profiled the non-null records across categorical features to verify domain validity and class cardinality.

### 4.1 Menu Items (`Item`)
- **Valid Observations:** 9,031 transactions (90.31% of dataset)
- **Null Records:** 969 (`np.nan`)
- **Unique Canonical Classes:** 8 items (0 noise classes)
  - `Coffee`: 1,228 orders (13.60%)
  - `Tea`: 1,199 orders (13.28%)
  - `Sandwich`: 1,187 orders (13.14%)
  - `Salad`: 1,164 orders (12.89%)
  - `Cake`: 1,140 orders (12.62%)
  - `Juice`: 1,112 orders (12.31%)
  - `Smoothie`: 1,006 orders (11.14%)
  - `Cookie`: 995 orders (11.02%)

### 4.2 Checkout Tender (`Payment Method`)
- **Valid Observations:** 6,822 transactions (68.22% of dataset)
- **Null Records:** 3,178 (`np.nan`)
- **Unique Canonical Classes:** 3 tender types
  - `Digital Wallet`: 3,731 orders (54.69% of valid tenders)
  - `Credit Card`: 1,551 orders (22.73% of valid tenders)
  - `Cash`: 1,540 orders (22.58% of valid tenders)

### 4.3 Order Fulfillment Mode (`Location`)
- **Valid Observations:** 6,039 transactions (60.39% of dataset)
- **Null Records:** 3,961 (`np.nan`)
- **Unique Canonical Classes:** 2 dining fulfillment types
  - `Takeaway`: 4,217 orders (69.83% of valid locations)
  - `In-store`: 1,822 orders (30.17% of valid locations)

---

## 5. Entity Integrity & Zero-Leakage Pipeline Verification

A dedicated verification engine (`verify_pseudonull_elimination()`) was executed against the standardized DataFrame to programmatically certify quality standards:

```python
verification = verify_pseudonull_elimination(df_clean)
assert verification['all_assertions_passed'] == True
```

### Verification Results:
- **Remaining Sentinel Tokens Across Entire DataFrame:** `0` (Zero token leakage)
- **Primary Key Uniqueness (`Transaction ID`):** `10,000 / 10,000` (100.0% entity integrity)
- **Primary Key Missingness:** `0` null values
- **Total Certified Null Cells:** `10,082` cells
- **Pipeline Certification Status:** **`PASSED`**

---

## 6. Diagnostic Visual Artifacts

Two publication-grade diagnostic figures were generated and serialized to `Reports/figures/`:

1. **`day4_pseudonull_normalization_waterfall.png`:**
   - Stacked horizontal decomposition chart illustrating the exact contribution of raw ingested `NaN`s, converted `'ERROR'` tokens, and converted `'UNKNOWN'` tokens for each attribute, displaying total post-normalization missingness percentages.
2. **`day4_standardized_categorical_profiles.png`:**
   - Multi-panel distribution plot depicting clean transaction frequencies and percentage shares for `Item` (8 menu products), `Payment Method` (3 tender channels), and `Location` (2 fulfillment modes) following sentinel elimination.

---

## 7. Milestone Progression Scorecard & Day 5 Handoff

| Metric Dimension | Day 1 Baseline | Day 2 Audit | Day 3 Quality Report | Day 4 Standardization | Day 4 Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Total Rows** | 10,000 | 10,000 | 10,000 | 10,000 | 100% records preserved |
| **Total Columns** | 8 | 8 | 8 | 8 | Schema structure maintained |
| **True Nulls (`NaN`)** | 6,826 | 6,826 | 6,826 | **10,082** | All disguised tokens unified |
| **Literal Pseudonulls** | Unaudited | 3,256 | 3,256 | **0** | **100% eliminated (3,256 converted)** |
| **Whitespace Cleanliness** | Unaudited | Unaudited | Unaudited | **100% Sanitized** | 0 leading/trailing whitespace |
| **Categorical Purity** | Polluted | Polluted | Polluted | **100% Canonical** | 8 items, 3 tenders, 2 locations |
| **Key Integrity** | Unverified | Verified | 100% Unique | **100% Unique** | 0 collisions, 0 nulls |
| **Pipeline Health Status** | Ingested | Cataloged | Baseline Certified | **Sanitized & Ready** | Ready for strict schema typing |

### Handoff to Day 5:
With all strings sanitized and pseudonulls mapped to IEEE `np.nan`, the dataset is now primed for **Day 5: Strict Schema Data Type Enforcement** (`datetime64[ns]` for transaction dates, `string` for keys, nullable `Int64` for quantities, and `float64` for unit prices and total spent).
