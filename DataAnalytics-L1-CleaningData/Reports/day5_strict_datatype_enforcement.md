# 🛡️ Cafe Sales – Day 5 Report: Strict Schema Data Type Enforcement & Validation
**Oasis Infobyte Data Analytics Internship · Task Level 1**  
**Milestone:** Day 5 of 18 — Strict Data Type Casting (`datetime64[ns]`, `string` ID, nullable `Int64`, `float64`)  
**Target Asset:** `Data/dirty_cafe_sales.csv` (10,000 rows × 8 columns)  
**Status:** Certified Day 5 Transformations Complete & Verified  

---

## 1. Executive Summary & Milestone Scope

On **Day 5**, the data cleaning pipeline transitioned from text normalization to **strict schema typing and dimensional domain validation**.

Following Day 4's sanitization, all extraneous whitespace was removed and 3,256 pseudonull string tokens (`'ERROR'`, `'UNKNOWN'`) were successfully converted to IEEE `np.nan`. However, all 8 attributes remained stored under Python's permissive `object` (string) dtype. In enterprise data engineering, leaving analytical columns typed as generic string objects introduces severe performance bottlenecks and architectural vulnerabilities:
- **Memory Overhead:** Object series store heap references and pointer metadata rather than raw contiguous memory blocks, resulting in unnecessary memory usage.
- **Computational Latency:** Arithmetic and statistical operations cannot utilize vectorized, SIMD-accelerated CPU instructions on string pointers.
- **Absence of Chronological Semantics:** String representations of dates (e.g. `'2023-09-08'`) cannot support calendar windowing, resampling, or chronological aggregations.
- **Silent Invariant Breakage:** Arbitrary text can enter numeric columns without compile-time or runtime schema rejection.

### Core Day 5 Engineering Achievements:
1. **Primary Key Typing (`string`):** Cast `Transaction ID` to pandas native `string` dtype, asserting 10,000 unique primary keys with 0 missing values conforming to regex pattern `^TXN_\d{7}$`.
2. **Temporal Domain Enforcement (`datetime64[ns]`):** Successfully parsed `Transaction Date` strings into ISO-8601 timestamps, validating that 100% of non-null transactions span all **365 contiguous calendar trading days of 2023** (`2023-01-01` to `2023-12-31`), mapping 460 missing records to canonical `NaT`.
3. **Discrete Integer Casting (`Int64`):** Cast `Quantity` to pandas nullable integer `Int64`, mapping missing values to `<NA>` while enforcing discrete purchase counts bounded between 1 and 5 units ($1 \le q \le 5$).
4. **Monetary Float Standardization (`float64`):** Enforced 64-bit IEEE float primitives across `Price Per Unit` ($1.00 to $5.00) and `Total Spent` ($1.00 to $25.00).
5. **Zero Coercion Data Loss ($\Delta = 0$):** Verified that exactly 0 valid entries failed parsing or suffered unintended truncation during conversion.
6. **14.73% Immediate Memory Reduction:** Reduced DataFrame footprint from **1,064.0 KB to 907.3 KB** prior to Day 6 categorical encoding.
7. **100% Assertion Sign-Off:** Automated verification suite certified zero schema violations and complete boundary fidelity.

---

## 2. Master Data Type Transformation & Schema Audit Matrix

The table below summarizes the before-and-after schema state, missing value representations, and validation outcomes across all 8 features:

| Attribute Name | Ingested Raw Dtype | Pre-Cast Dtype (Day 4) | Enforced Operational Dtype (Day 5) | Pre-Cast Nulls | Post-Cast Nulls | Unparseable Coerced ($\Delta$) | Null Representation | Validation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Transaction ID`** | `object` | `object` (`str`) | **`string`** | 0 | 0 | **0** | None (0.0% nulls) | **PASSED (100% Unique)** |
| **`Item`** | `object` | `object` (`str`) | **`object` (`str`)** | 969 | 969 | **0** | `np.nan` | **READY (Day 6 Category)** |
| **`Quantity`** | `object` | `object` (`str`) | **`Int64`** (nullable) | 479 | 479 | **0** | `<NA>` | **PASSED (Range: 1–5)** |
| **`Price Per Unit`** | `object` | `object` (`str`) | **`float64`** | 533 | 533 | **0** | `np.nan` | **PASSED (Menu Prices)** |
| **`Total Spent`** | `object` | `object` (`str`) | **`float64`** | 502 | 502 | **0** | `np.nan` | **PASSED (Range: $1–$25)** |
| **`Payment Method`** | `object` | `object` (`str`) | **`object` (`str`)** | 3,178 | 3,178 | **0** | `np.nan` | **READY (Day 6 Category)** |
| **`Location`** | `object` | `object` (`str`) | **`object` (`str`)** | 3,961 | 3,961 | **0** | `np.nan` | **READY (Day 6 Category)** |
| **`Transaction Date`** | `object` | `object` (`str`) | **`datetime64[ns]`** | 460 | 460 | **0** | `NaT` | **PASSED (365 Days 2023)** |
| **TOTALS** | — | — | — | **10,082** | **10,082** | **0** | — | **Zero Coercion Loss** |

### Mathematical Proof of Schema Integrity:
$$\forall j \in \{1, \dots, 8\}, \quad \text{Nulls}_{\text{Post-Cast}, j} - \text{Nulls}_{\text{Pre-Cast}, j} \equiv 0$$
$$\sum_{j=1}^{8} \Delta_j = 0$$

Every valid string value was successfully parsed into its respective operational primitive with zero fallback errors.

---

## 3. Detailed Attribute-by-Attribute Engineering Rationale

### 3.1 Primary Key: `Transaction ID` $\rightarrow$ `string`
- **Technical Selection:** pandas `string` (backed by Python strings / Arrow).
- **Rationale:** Prevents unintended object manipulation and clearly identifies this column as non-numeric identifier metadata.
- **Integrity Audit:** 10,000 / 10,000 unique values; 0 nulls; exact pattern match against `^TXN_\d{7}$`.

### 3.2 Discrete Quantity: `Quantity` $\rightarrow$ `Int64`
- **Technical Selection:** pandas nullable integer `Int64`.
- **Rationale:** Traditional NumPy `int64` cannot store missing values without coercing the entire column into floating-point numbers (`1.0`, `2.0`), which distorts discrete item purchase semantics. `Int64` natively preserves discrete integer values alongside `<NA>`.
- **Validation Bounds:** Minimum = 1 unit, Maximum = 5 units. Discrete frequency distribution:
  - 1 unit: 1,939 orders (20.37%)
  - 2 units: 1,911 orders (20.07%)
  - 3 units: 1,911 orders (20.07%)
  - 4 units: 1,894 orders (19.89%)
  - 5 units: 1,866 orders (19.60%)

### 3.3 Monetary Floats: `Price Per Unit` & `Total Spent` $\rightarrow$ `float64`
- **Technical Selection:** Standard 64-bit IEEE double-precision floating-point (`float64`).
- **Rationale:** Financial values require decimal support. Unit prices represent dollar amounts with half-dollar increments (e.g., $1.50).
- **Price Domain Invariant:** Valid prices strictly conform to the 6 menu price tiers: $\{\$1.00, \$1.50, \$2.00, \$3.00, \$4.00, \$5.00\}$.
- **Total Spent Bounds:** Range strictly bounded between $\$1.00$ (minimum order: $1 \times \$1.00$) and $\$25.00$ (maximum order: $5 \times \$5.00$).

### 3.4 Temporal Dimension: `Transaction Date` $\rightarrow$ `datetime64[ns]`
- **Technical Selection:** Contiguous nanosecond datetime primitive (`datetime64[ns]`).
- **Rationale:** Unlocks native temporal operations, chronological sorting, time-series resampling, and calendar period derivations.
- **Temporal Domain Bounds:**
  - Earliest Transaction: `2023-01-01`
  - Latest Transaction: `2023-12-31`
  - Unique Trading Days: **365 / 365 days** (100% calendar completeness)
  - Daily Volume: Mean = 26.14 orders/day (Min = 14, Max = 40)
  - Missing Values: 460 transactions mapped to canonical `NaT` (ready for chronological imputation in Day 7–10).

---

## 4. Performance & Memory Footprint Audit

Transitioning from general `object` pointers to compact primitives produces an immediate performance and memory optimization:

```text
================ MEMORY FOOTPRINT BENCHMARK ================
Pre-Cast (Day 4 Normalized Objects)   : 1,064.00 KB
Post-Cast (Day 5 Strict Primitive DF)  :   907.30 KB
Net Memory Reduction                  :  -156.70 KB (-14.73%)
=============================================================
```

> [!NOTE]
> On **Day 6**, categorical encoding of `Item`, `Payment Method`, and `Location` will yield an additional ~40–50% memory compaction, bringing total pipeline memory savings to over 75%.

---

## 5. Automated Pipeline Verification & Quality Assertions

The automated verification engine (`verify_type_integrity()`) executed the full test battery:

```python
verif = verify_type_integrity(df_typed, df_clean)
assert verif['all_assertions_passed'] == True
```

### Verification Battery Execution Results:
- `is_tx_id_string`: **`True`**
- `is_date_datetime`: **`True`**
- `is_qty_int64`: **`True`**
- `is_price_float64`: **`True`**
- `is_total_float64`: **`True`**
- `unintended_coercions`: **`0`**
- `pk_unique_count`: **`10,000`**
- `pk_null_count`: **`0`**
- `pk_valid_pattern`: **`True`** (`^TXN_\d{7}$`)
- `qty_valid_range`: **`True`** ($1 \le q \le 5$)
- `qty_discrete_integers`: **`True`**
- `prices_canonical`: **`True`** ($\subset \{\$1.00, \$1.50, \$2.00, \$3.00, \$4.00, \$5.00\}$)
- `totals_valid_range`: **`True`** ($\$1.00 \le \text{Total} \le \$25.00$)
- `date_in_2023`: **`True`** (`2023-01-01` to `2023-12-31`)
- `unique_trading_days`: **`365`**
- `pipeline_status`: **`PASSED`**

---

## 6. Diagnostic Visual Artifacts

Two publication-grade diagnostic figures have been generated and serialized to `Reports/figures/`:

1. **`day5_datatype_transformation_matrix.png`:**
   - Architectural schema scorecard detailing pre-cast dtypes, post-cast enforced primitives, null representations (`<NA>`, `NaT`, `np.nan`), missing counts, domain constraints, and validation badges.
2. **`day5_numerical_and_temporal_profiles.png`:**
   - Multi-panel diagnostic suite featuring:
     - Discrete purchase count bar chart for `Quantity` ($1 \dots 5$ units).
     - Parallel boxplots illustrating the financial scales of `Price Per Unit` vs. `Total Spent`.
     - 365-day continuous time-series trend of daily transactions across 2023 with a 7-day rolling average.

---

## 7. Milestone Progression Scorecard & Day 6 Strategic Handoff

| Metric Dimension | Day 1 Baseline | Day 2 Anomaly Audit | Day 3 Quality Report | Day 4 Standardization | Day 5 Type Enforcement | Cumulative Progress |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Total Rows** | 10,000 | 10,000 | 10,000 | 10,000 | **10,000** | 100% preservation |
| **Total Columns** | 8 | 8 | 8 | 8 | **8** | Canonical schema intact |
| **True Missing Cells** | 6,826 | 6,826 | 6,826 | 10,082 | **10,082** | Consolidated baseline |
| **Disguised Pseudonulls** | Unaudited | 3,256 | 3,256 | 0 | **0** | 100% eliminated |
| **Whitespace Quality** | Unaudited | Unaudited | Unaudited | 100% Sanitized | **100% Sanitized** | 0 text anomalies |
| **Operational Typing** | 0% (All `object`) | 0% (All `object`) | 0% (All `object`) | 0% (All `object`) | **62.5% Strictly Typed** | ID, Date, Qty, Price, Total |
| **Temporal Index Ready** | No (`str`) | No (`str`) | No (`str`) | No (`str`) | **Yes (`datetime64[ns]`)** | 365 days validated |
| **RAM Footprint** | ~1,064 KB | ~1,064 KB | ~1,064 KB | ~1,064 KB | **907.3 KB (-14.73%)** | Memory optimized |
| **Pipeline Status** | Raw Ingestion | Cataloged | Baseline Signed Off | Sanitized | **Strictly Typed & Validated** | Ready for Categoricals |

### Strategic Handoff to Day 6:
With identifiers, temporal records, and numeric features strictly typed and verified, the next scheduled milestone is **Day 6: Categorical Encoding (`Item`, `Payment Method`, `Location`) and Comprehensive Memory Footprint Auditing**.
