# ☕ Cafe Sales – Day 12 Report: Statistical Numerical Outlier Detection & Moment Profiling
**Oasis Infobyte Data Analytics Internship · Task Level 1**  
**Milestone:** Day 12 of 18 — Dual-Method Statistical Outlier Detection (Tukey IQR Fences & 3-Sigma Z-Score), Distributional Moment Profiling & Candidate Anomaly Identification  
**Target Asset:** `Data/dirty_cafe_sales.csv` (10,000 rows × 8 columns)  
**Status:** Certified Day 12 Statistical Profiling & 100.00% Zero-Data-Loss Verification  

---

## 1. Executive Summary & Milestone Scope

On **Day 12**, the cafe sales data cleaning and transformation pipeline transitioned into statistical numerical distribution modeling and outlier detection. Following Day 11's mathematical proof of zero transaction duplication and certification of 100.00% universal cell completeness (80,000 / 80,000 certified non-null cells), Day 12 audited the distributional geometry of all continuous and discrete numeric features: `Quantity`, `Price Per Unit`, and `Total Spent`.

In commercial retail analytics, distinguishing between **corrupted rogue data** (such as decimal transposition errors, sensor recording glitches, or negative quantities) and **legitimate boundary customer transactions** (such as high-volume orders of premium catalog items) is paramount. Applying naive outlier rejection algorithms blindly purges high-spending customers and distorts financial reporting.

To establish defensible statistical governance, Day 12 deployed a **Dual-Paradigm Outlier Detection Engine** combining non-parametric **Tukey's Fences** with parametric **3-Sigma Standardized Z-Scores**, accompanied by **Higher-Order Distributional Moment Profiling**:

```text
+---------------------------------------------------------------------------------------------------------------+
|                                    DAY 12 STATISTICAL OUTLIER DETECTION SCORECARD                             |
+---------------------------------------------------------------------------------------------------------------+
|  Metric Dimension                   Observed Metric       Target / Benchmark      Engineering Assessment      |
|  ----------------                   ---------------       ------------------      ----------------------      |
|  Dataset Record Retention           10,000 / 10,000       100.0% Retention        Zero Destructive Data Loss  |
|  Canonical Features Intact          8 / 8 Columns         100.0% Schema Match     Certified Schema Integrity  |
|  Master Cell Completeness           80,000 / 80,000       100.00% Clean Cells     Zero Missing Cells Intact   |
|  Quantity Outliers (IQR / Z)        0 / 0 Records         Domain: [1, 5] Units    100.0% Valid Menu Quantities|
|  Price Per Unit Outliers (IQR / Z)  0 / 0 Records         Domain: [$1.00, $5.00]  100.0% Canonical Menu Match |
|  Total Spent Range                  $1.00 – $25.00        Domain Bounded          Discrete Geometry Match     |
|  Total Spent Tukey Upper Fence      $24.00 (Q3 + 1.5*IQR) Q3=$12, IQR=$8          Mathematically Verified     |
|  Total Spent Mild IQR Outliers      271 Records (2.71%)   All at $25.00           Candidate Cohort Isolated   |
|  Total Spent Extreme Outliers       0 Records (0.00%)     Threshold: > $36.00     Zero Rogue Escapes Confirmed|
|  Total Spent 3-Sigma Z-Outliers     0 Records (0.00%)     Threshold: |Z| > 3.0    Max |Z| = 2.6722 < 3.0      |
|  Candidate Cohort Homogeneity       100.0% Salad (5 units)$5.00 Unit Price        Single Structural Profile   |
|  Candidate Accounting Identity      0 Violations          5 * $5.00 == $25.00     100.0% Fidelity ($0.000000) |
|  Commercial Revenue at Stake        $6,775.00 USD         $89,395.00 Total        7.58% Gross Store Revenue   |
|  Total Spent Skewness (g1)          +0.8193               Positive Right-Skew     Natural Retail Spend Curve  |
|  Total Spent Excess Kurtosis (g2)   -0.1540               Mesokurtic / Thin Tail  Absence of Heavy Rogue Tail |
|  Automated QA Assertions Passed     19 / 19 Assertions    100% Sign-Off           Production Certified        |
+---------------------------------------------------------------------------------------------------------------+
```

### Core Day 12 Engineering Achievements:
1. **Zero Outliers in Primary Transactional Primitives:** Verified that both `Quantity` (range $[1, 5]$ units) and `Price Per Unit` (range $[\$1.00, \$5.00]$ USD) exhibit **0 Tukey IQR outliers** and **0 3-Sigma Z-score outliers**, with 100% compliance with catalog constraints.
2. **Dual-Model Statistical Fence Auditing:** Evaluated non-parametric Tukey's Fences ($1.5 \times \text{IQR}$ and $3.0 \times \text{IQR}$) against parametric 3-Sigma boundaries across all 10,000 transactions.
3. **Isolation of 271 Boundary Candidate Orders:** Identified exactly 271 transactions ($2.71\%$) flagged as mild statistical outliers by Tukey's $1.5 \times \text{IQR}$ upper fence ($\text{Total Spent} > \$24.00$). All 271 records possess a uniform spend value of **\$25.00**.
4. **Complete Absence of Extreme Rogue Values:** Confirmed **0 extreme outliers** exceeding Tukey's outer fence ($Q_3 + 3.0 \times \text{IQR} = \$36.00$) and **0 observations exceeding the 3-sigma threshold ($|Z| > 3.0$)**, proving that no decimal corruptions (e.g., $\$250.00$) or arbitrary entry errors exist.
5. **Structural Homogeneity of Flagged Cohort:** Audited the underlying features of all 271 flagged transactions. Confirmed that **100% represent purchases of 5 units of Salad at \$5.00 per unit**, perfectly satisfying the accounting invariant $5 \times \$5.00 = \$25.00$ with zero discrepancy ($\Delta = \$0.000000$).
6. **Commercial Exposure Quantification:** Measured that the flagged candidate cohort accounts for **\$6,775.00 USD (7.58% of gross annual store revenue)**, providing the empirical foundation for Day 13's formal retention proofs.
7. **Distributional Moment Profiling:** Established baseline moments for `Total Spent`: mean $\$8.94$, median $\$8.00$, mode $\$4.00$, positive right-skewness ($g_1 = +0.8193$), and platykurtic tail profile ($g_2 = -0.1540$).

---

## 2. Theoretical Foundations: Dual-Paradigm Outlier Detection

### 2.1 Non-Parametric Tukey's Fences (Interquartile Range)
Tukey's method relies exclusively on rank-order quartiles, conferring complete insensitivity to extreme tail contamination and non-normality:

$$\text{IQR} = Q_3 - Q_1$$

Where $Q_1$ is the 25th percentile and $Q_3$ is the 75th percentile of the empirical distribution. Two demarcation fences are established:

$$\text{Inner Fences (Mild Outliers):} \quad [Q_1 - 1.5 \times \text{IQR}, \quad Q_3 + 1.5 \times \text{IQR}]$$

$$\text{Outer Fences (Extreme Outliers):} \quad [Q_1 - 3.0 \times \text{IQR}, \quad Q_3 + 3.0 \times \text{IQR}]$$

- **Mild Outliers:** Data points lying between the inner and outer fences ($1.5 \times \text{IQR} < |x - Q| \le 3.0 \times \text{IQR}$). In asymmetric business distributions, these frequently represent high-capacity legitimate activity.
- **Extreme Outliers:** Data points lying strictly outside the outer fences ($|x - Q| > 3.0 \times \text{IQR}$). In automated data pipelines, extreme outliers almost invariably signify ingestion failure or system corruption.

### 2.2 Parametric Standardized 3-Sigma Z-Score
The Z-score models the distance between each sample observation $x_i$ and the sample arithmetic mean $\mu$, measured in standard deviation units $\sigma$:

$$Z_i = \frac{x_i - \mu}{\sigma}$$

Under Chebyshev's inequality and the empirical 3-sigma rule for unimodal distributions:

$$P(|Z| \ge 3.0) \le \frac{1}{3^2} \approx 11.1\% \quad (\text{General}), \quad P(|Z| \ge 3.0) \approx 0.27\% \quad (\text{Gaussian})$$

Observations where $|Z_i| > 3.0$ are classified as parametric outliers.

### 2.3 Methodological Synthesis & Asymmetry Rationale
Consumer spend distributions in retail point-of-sale environments are inherently non-Gaussian. Transaction totals are bounded below by zero (or minimum catalog price) and exhibit a long right-tail of multi-item purchases.

Consequently, Tukey's $1.5 \times \text{IQR}$ rule acts as a sensitive diagnostic scanner for boundary behavior, while the 3-Sigma Z-score and $3.0 \times \text{IQR}$ outer fences prevent false-positive over-pruning of legitimate high-value commerce.

---

## 3. Empirical Statistical Outlier Audit Matrix

The table below compiles the comprehensive parametric and non-parametric statistical metrics computed across all 10,000 transactions for each numeric dimension:

| Attribute | Observed Min | Observed Max | Sample Mean ($\mu$) | Sample Std ($\sigma$) | Median ($Q_2$) | 25th Pct ($Q_1$) | 75th Pct ($Q_3$) | $\text{IQR}$ | Tukey Inner Fence | Tukey Outer Fence | Tukey Mild Outliers | Tukey Mild (%) | Tukey Extreme Outliers | 3-Sigma Outliers ($|Z| > 3$) | Max Observed $|Z|$ | Operational Diagnostic Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Quantity`** | 1.00 | 5.00 | 3.0278 | 1.4211 | 3.0000 | 2.0000 | 4.0000 | 2.0000 | [-1.00, 7.00] | [-4.00, 10.00] | **0** | **0.00%** | **0** | **0** | 1.4270 | **In-Domain (100% Valid)** |
| **`Price Per Unit`** | \$1.00 | \$5.00 | \$2.9474 | \$1.2798 | \$3.0000 | \$2.0000 | \$4.0000 | \$2.0000 | [-\$1.00, \$7.00] | [-\$4.00, \$10.00] | **0** | **0.00%** | **0** | **0** | 1.6038 | **In-Domain (100% Valid)** |
| **`Total Spent`** | \$1.00 | \$25.00 | \$8.9395 | \$6.0101 | \$8.0000 | \$4.0000 | \$12.0000 | \$8.0000 | [-\$8.00, **\$24.00**] | [-\$20.00, **\$36.00**] | **271** | **2.71%** | **0** | **0** | **2.6722** | **Mild Boundary Candidate** |

### Analytical Deductions from Audit Matrix:
1. **Quantity Boundedness:** The empirical distribution of `Quantity` spans exactly $1 \dots 5$ units. Because the inner fences are $[-1.0, 7.0]$, all 10,000 records fall safely within bounds. Max $|Z| = 1.4270 \ll 3.0$.
2. **Menu Pricing Invariance:** Unit prices span exactly $\$1.00 \dots \$5.00$. With inner fences at $[-\$1.00, \$7.00]$, zero pricing anomalies exist. Max $|Z| = 1.6038 \ll 3.0$.
3. **Total Spent Inner Fence Breach:** For `Total Spent`, $Q_1 = \$4.00$ and $Q_3 = \$12.00$, yielding $\text{IQR} = \$8.00$. The Tukey upper inner fence is calculated as:
   $$\text{Upper Inner Fence} = \$12.00 + 1.5 \times \$8.00 = \mathbf{\$24.00}$$
   Exactly **271 records** exhibit $\text{Total Spent} = \$25.00 > \$24.00$, qualifying them as mild statistical outlier candidates.
4. **Complete Absence of Extreme Outliers:**
   $$\text{Upper Outer Fence} = \$12.00 + 3.0 \times \$8.00 = \mathbf{\$36.00}$$
   No transaction in the dataset exceeds $\$36.00$ (max observed is $\$25.00$). The extreme outlier count is strictly **0 (0.00%)**.
5. **Parametric Normal Range Compliance:** The maximum standardized Z-score for $\text{Total Spent} = \$25.00$ is:
   $$Z_{\text{max}} = \frac{\$25.00 - \$8.9395}{\$6.0101} = \mathbf{+2.6722} < 3.0000$$
   Under the 3-sigma criterion, **zero records** are flagged as outliers.

---

## 4. Higher-Order Distributional Moment Analysis

To characterize the underlying probability density shapes beyond central tendency and variance, higher-order standardized moments were computed:

| Attribute | Arithmetic Mean ($\mu$) | Std Dev ($\sigma$) | Fisher-Pearson Skewness ($g_1$) | Skewness Shape Interpretation | Excess Kurtosis ($g_2$) | Tail Profile Interpretation | D'Agostino-Pearson $K^2$ | Normality Test $p$-value |
| :--- | :---: | :---: | :---: | :--- | :---: | :--- | :---: | :---: |
| **`Quantity`** | 3.0278 | 1.4211 | -0.0076 | Approximately Symmetric | -1.3143 | Platykurtic (Light Uniform Tails) | 56,842.43 | $0.0000 \times 10^0$ |
| **`Price Per Unit`** | 2.9474 | 1.2798 | +0.0076 | Approximately Symmetric | -1.1586 | Platykurtic (Light Uniform Tails) | 6,157.05 | $0.0000 \times 10^0$ |
| **`Total Spent`** | 8.9395 | 6.0101 | **+0.8193** | **Moderately Right-Skewed** | **-0.1540** | **Mesokurtic / Light Tail** | 881.04 | $4.84 \times 10^{-192}$ |

### Moment Interpretations:
- **Skewness ($g_1 = +0.8193$):** Total spending exhibits positive right-skewness. In retail purchasing, the vast majority of tickets represent 1–2 items ($4.00 – $8.00), while larger multi-item baskets taper toward $\$20.00 – \$25.00$. This matches authentic macroeconomic retail behavior.
- **Excess Kurtosis ($g_2 = -0.1540$):** Total spending has slightly negative excess kurtosis ($g_2 < 0$), characterizing a **platykurtic** distribution with thinner tails than a standard normal distribution. This is critical: if runaway outliers or decimal entry corruptions existed, $g_2$ would be heavily positive (leptokurtic). The negative kurtosis proves that the upper tail is strictly bounded.

---

## 5. In-Depth Forensic Investigation: The \$25.00 Transaction Cohort

We performed an exhaustive forensic examination of the 271 transactions flagged by Tukey's inner upper fence:

```text
+---------------------------------------------------------------------------------------------------------------+
|                                    CANDIDATE COHORT FORENSIC PROFILE ($25.00 ORDERS)                          |
+---------------------------------------------------------------------------------------------------------------+
|  Diagnostic Dimension               Observed Profile                  Engineering Audit Finding               |
|  --------------------               ----------------                  -------------------------               |
|  Total Records Flagged              271 transactions (2.71%)          Exclusively Total Spent > $24.00        |
|  Recorded Total Spent               $25.00 across all 271 rows        100.0% Uniform Price Point              |
|  Purchased Catalog Item             'Salad' across 271 rows (100.0%)  0 Other Products Present                |
|  Purchased Basket Quantity          5 units across 271 rows (100.0%)  0 Other Quantities Present              |
|  Unit Price Charged                 $5.00 across 271 rows (100.0%)    Exact Menu Price Invariant Matched      |
|  Accounting Identity Validation     5 * $5.00 == $25.00               0 Discrepancies (Delta = $0.000000)     |
|  Temporal Cadence Across 2023       12 Calendar Months                Steady: Mean 22.6 orders / month        |
|  Minimum Monthly Frequency          13 orders (January 2023)          Normal Operational Baseline             |
|  Maximum Monthly Frequency          32 orders (March 2023)            Seasonal Retail Fluctuation             |
|  Fulfillment Split                  Takeaway: 71.22% | In-store: 28.78% Matches Storewide (69.8% / 30.2%)     |
|  Tender Channel Split               Wallet: 54.6% | Card: 24.0% | Cash: Matches Storewide (54.7% / 22.7% / 22.6%)
|  Total Commercial Revenue           $6,775.00 USD                     7.58% of Annual Store Sales             |
+---------------------------------------------------------------------------------------------------------------+
```

### Key Forensic Findings:
1. **Mathematical Invariant Perfection:** Every single one of the 271 orders represents an order of **5 Salads at \$5.00 each**. Because Salad is the cafe's highest-priced menu offering (\$5.00) and 5 is the maximum allowable order quantity, $\$25.00$ represents the theoretical maximum boundary of the catalog's discrete pricing lattice:
   $$\max(\text{Spend}) = \max(\text{Quantity}) \times \max(\text{Price}) = 5 \times \$5.00 = \mathbf{\$25.00}$$
2. **Temporal Uniformity:** These 271 orders did not occur in an isolated burst or system batch anomaly. They occurred steadily throughout all 365 days of 2023, averaging $\approx 22.6$ transactions per month with a standard deviation of $\pm 5.1$.
3. **Operational Consistency:** The payment method breakdown (54.6% Digital Wallet, 24.0% Credit Card, 21.4% Cash) and fulfillment location breakdown (71.2% Takeaway, 28.8% In-store) mirror the overall cafe operational baselines perfectly ($|\Delta| < 1.4\%$).

---

## 6. Risk Analysis of Outlier Treatment Paradigms

Before finalizing engineering decisions on Day 13, Day 12 evaluated the three canonical outlier treatment options:

```text
  +-------------------------------------------------------------------------------------------------+
  |                       CANDIDATE OUTLIER TREATMENT PARADIGM RISK ASSESSMENT                      |
  +-------------------------------------------------------------------------------------------------+
                                                  │
         ┌────────────────────────────────────────┼────────────────────────────────────────┐
         ▼                                        ▼                                        ▼
┌──────────────────┐                     ┌──────────────────┐                     ┌──────────────────┐
│  WINSORIZATION   │                     │    ROW PURGING   │                     │ DOMAIN RETENTION │
│   (Capping at    │                     │   (Dropping 271  │                     │   (Certifying    │
│     $24.00)      │                     │     Records)     │                     │   100% Invariant)│
└────────┬─────────┘                     └────────┬─────────┘                     └────────┬─────────┘
         │                                        │                                        │
         ▼                                        ▼                                        ▼
• Violates accounting identity:          • Severe data loss: 271 orders           • 100% accounting identity
  5 * $5.00 == $25.00 != $24.00            destroyed (2.71% of dataset)             preserved (Delta = $0.00)
• Injects $271.00 ledger error           • Destroys $6,775.00 USD in revenue      • Protects $6,775.00 USD
• Breaks financial reconciliation          (7.58% of gross annual turnover)         in verified annual sales
• Creates artificial pricing distortion  • Distorts AOV and customer velocity     • Reflects valid POS boundary
```

1. **Paradigm A: Winsorization (Capping to \$24.00 Upper Fence):**
   - *Consequence:* Capping `Total Spent` to $\$24.00$ while retaining $\text{Quantity} = 5$ and $\text{Price Per Unit} = \$5.00$ introduces a fatal contradiction into the database:
     $$\Delta = |\text{Total Spent} - (\text{Quantity} \times \text{Price})| = |\$24.00 - \$25.00| = \mathbf{\$1.00 \text{ discrepancy per row}}$$
   - Over 271 records, this injects **\$271.00 in unresolvable accounting errors** and destroys auditability.
2. **Paradigm B: Record Deletion / Truncation:**
   - *Consequence:* Deleting the 271 records purges **2.71% of the company's transactions** and **\$6,775.00 USD in gross revenue (7.58% of turnover)**.
   - Deleting high-value customers introduces severe **survivorship bias**, causing management to systematically underestimate peak order values and premium product demand.
3. **Paradigm C: Domain-Informed Statistical Retention:**
   - *Consequence:* Retaining the records preserves 100% accounting fidelity ($\Delta = \$0.000000$), protects \$6,775.00 USD in legitimate commercial turnover, and acknowledges that $\$25.00$ is a valid structural boundary point of the retail menu rather than a technical error.

---

## 7. Automated Quality Assurance & Assertion Verification Matrix

The automated assertion suite `verify_day12_outlier_integrity()` was executed across 19 critical checkpoints:

| Assertion Check ID | Verification Dimension | Target Criterion | Observed Value | Verification Status |
| :---: | :--- | :--- | :---: | :---: |
| **`C01`** | **Record Retention** | $\text{Len}(\text{df}_{d12}) == 10,000$ | **10,000 Records** | **PASSED (100.0%)** |
| **`C02`** | **Schema Preservation** | $\text{Cols}(\text{df}_{d12}) == 8$ canonical features | **8 Columns Intact** | **PASSED (100.0%)** |
| **`C03`** | **Universal Completeness** | $\text{Cells}_{\text{non-null}} == 80,000$ (0 nulls) | **80,000 Clean Cells** | **PASSED (100.0%)** |
| **`C04`** | **Quantity Bounds** | $\min \ge 1 \land \max \le 5$ | **1 to 5 Units** | **PASSED (100.0%)** |
| **`C05`** | **Quantity IQR Outliers** | $\text{Count}(\text{Outliers}_{\text{IQR}}) == 0$ | **0 Outliers** | **PASSED (100.0%)** |
| **`C06`** | **Price Bounds** | $\min \ge \$1.00 \land \max \le \$5.00$ | **\$1.00 to \$5.00** | **PASSED (100.0%)** |
| **`C07`** | **Price IQR Outliers** | $\text{Count}(\text{Outliers}_{\text{IQR}}) == 0$ | **0 Outliers** | **PASSED (100.0%)** |
| **`C08`** | **Total Spent Bounds** | $\min \ge \$1.00 \land \max \le \$25.00$ | **\$1.00 to \$25.00** | **PASSED (100.0%)** |
| **`C09`** | **Tukey Upper Fence** | $Q_3 + 1.5 \times \text{IQR} == \$24.00$ | **\$24.00 Exactly** | **PASSED (100.0%)** |
| **`C10`** | **Mild IQR Outliers** | $\text{Count}(\text{Spend} > \$24.00) == 271$ | **271 Records (2.71%)** | **PASSED (100.0%)** |
| **`C11`** | **Zero Extreme IQR Outliers**| $\text{Count}(\text{Spend} > \$36.00) == 0$ | **0 Records (0.00%)** | **PASSED (100.0%)** |
| **`C12`** | **Zero 3-Sigma Outliers** | $\text{Count}(|Z| > 3.0) == 0$ across all cols | **0 Records (0.00%)** | **PASSED (100.0%)** |
| **`C13`** | **Maximum Standardized Z** | $\max(|Z|) < 3.0$ | **2.6722 for Spend** | **PASSED (100.0%)** |
| **`C14`** | **Cohort Item Homogeneity** | $\text{Item} == \text{'Salad'}$ for all flagged | **100.0% Salad** | **PASSED (100.0%)** |
| **`C15`** | **Cohort Qty Homogeneity** | $\text{Quantity} == 5$ for all flagged | **100.0% 5 Units** | **PASSED (100.0%)** |
| **`C16`** | **Cohort Price Homogeneity** | $\text{Price} == \$5.00$ for all flagged | **100.0% \$5.00** | **PASSED (100.0%)** |
| **`C17`** | **Cohort Accounting Match** | $\text{Qty} \times \text{Price} == \text{Total}$ for all flagged | **$\Delta = \$0.000000$** | **PASSED (100.0%)** |
| **`C18`** | **Global Accounting Match** | $\text{Qty} \times \text{Price} == \text{Total}$ across all 10k | **$\Delta = \$0.000000$** | **PASSED (100.0%)** |
| **`C19`** | **Total Spent Positive Skew** | $\text{Skewness}(g_1) > 0.5$ | **+0.8193 Right-Skewed**| **PASSED (100.0%)** |
| **`ALL`** | **Master Day 12 Sign-Off** | All 19 assertions certified | **19 / 19 Passed** | **PASSED (100.0%)** |

---

## 8. Diagnostic Visual Artifacts

Two publication-grade diagnostic visual figures (300 DPI) were generated and serialized to `Reports/figures/`:

1. **`Reports/figures/day12_statistical_outlier_detection_fences.png`**  
   *A quad layout illustrating:*
   - **Panel 1: Tukey's Fences Boxplot for Total Spent:** Displays quartiles ($Q_1=\$4.00$, $\text{Median}=\$8.00$, $Q_3=\$12.00$), the inner fence threshold at $\$24.00$, the outer fence at $\$36.00$, and the 271 candidate points clustered at $\$25.00$.
   - **Panel 2: Standardized 3-Sigma Z-Score Distributions:** Kernel density curves for `Quantity`, `Price Per Unit`, and `Total Spent`, confirming all observations fall strictly within $[-3.0, +3.0]$.
   - **Panel 3: Spend Geometry & Deterministic Pricing Rays:** Scatter plot mapping $\text{Quantity} \times \text{Price}$ to $\text{Total Spent}$, demonstrating that all records lie strictly upon linear rays and highlighting the boundary cluster at $(5, \$25.00)$.
   - **Panel 4: Outlier Frequency Comparison Bar Chart:** Comparative scorecard contrasting Tukey 1.5*IQR Mild (271 flags) against Tukey 3.0*IQR Extreme (0 flags) and 3-Sigma Z-Score (0 flags).

2. **`Reports/figures/day12_distributional_moments_and_spend_tails.png`**  
   *A quad layout illustrating:*
   - **Panel 1: Empirical Probability Density & Moments of Total Spent:** Histogram and KDE with overlaid mean (\$8.94), median (\$8.00), and mode (\$4.00), highlighting right-skewness ($g_1 = +0.8193$).
   - **Panel 2: Normal Quantile-Quantile (Q-Q) Plot:** Compares sample spend quantiles against theoretical standard normal quantiles, evidencing discrete step pricing and a bounded upper tail.
   - **Panel 3: Temporal Uniformity of \$25.00 Orders Across 2023:** Monthly distribution bar chart showing consistent order occurrence across all 12 calendar months (mean 22.6 orders/month, zero cluster spikes).
   - **Panel 4: Operational Mix Heatmap for Flagged Orders:** Crosstab heatmap of `Location` by `Payment Method`, proving alignment with overall cafe operating splits.

---

## 9. Business & Commercial Financial Impact Analysis

### 9.1 Protection of \$6,775.00 USD in Annual Sales
The 271 flagged transactions represent:
- **Commercial Volume:** 1,355 units of fresh Salads sold.
- **Monetary Turnover:** **\$6,775.00 USD**, constituting **7.58% of total gross cafe revenue (\$89,395.00 USD)**.
- **Average Order Contribution:** Each order generates **\$25.00**, representing nearly **$3\times$ the storewide Average Order Value (\$8.94)**.

Treating these transactions as illegitimate outliers to be removed or capped would inflict severe harm on commercial reporting by erasing the business's most valuable individual ticket transactions.

---

## 10. Master Entity, Completeness & Quality Lifecycle Matrix (Days 1–12)

The master table below summarizes the cumulative progression of dataset health, cleanliness, and structural integrity across the entire project lifecycle through Day 12:

| Milestone Stage | Target Scope | Retained Records | Non-Null Cells | Universal Completeness (%) | Duplicates Purged | Key Integrity | Accounting Violations | Outlier Status | Cumulative Revenue |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Day 1: Ingest** | Raw CSV inspection | 10,000 | 66,944 | 83.68% | 0 | Unverified | Unverified | Unverified | \$89,395.00 |
| **Day 4: Clean** | Text standardization | 10,000 | 66,944 | 83.68% | 0 | Valid Format | Unverified | Unverified | \$89,395.00 |
| **Day 5: Types** | Strict schema casting | 10,000 | 66,944 | 83.68% | 0 | Strict Strings | Unverified | Unverified | \$89,395.00 |
| **Day 6: Memory** | Categorical compression | 10,000 | 66,944 | 83.68% | 0 | 10,000 Unique | Unverified | Unverified | \$89,395.00 |
| **Day 7: Prices** | Menu invariant imputation | 10,000 | 69,918 | 87.40% | 0 | 10,000 Unique | In Progress | Unverified | \$89,395.00 |
| **Day 8: Math** | Accounting recovery | 10,000 | 71,373 | 89.22% | 0 | 10,000 Unique | **0 Violations** | Unverified | \$89,395.00 |
| **Day 9: Categories**| Distribution preservation | 10,000 | 79,540 | 99.43% | 0 | 10,000 Unique | 0 Violations | Unverified | \$89,395.00 |
| **Day 10: Dates** | Chronological fill | 10,000 | **80,000** | **100.00%** | 0 | 10,000 Unique | 0 Violations | Unverified | \$89,395.00 |
| **Day 11: Dedup** | Multi-tier entity audit | 10,000 | **80,000** | **100.00%** | **0** | **100% Unique** | **0 Violations** | Unverified | \$89,395.00 |
| **Day 12: Outliers**| Statistical dual-modeling| **10,000** | **80,000** | **100.00%** | **0** | **100% Unique** | **0 Violations** | **271 Mild / 0 Extreme**| **\$89,395.00 (100%)**|

---

## 11. Strategic Handoff to Day 13

With numerical outliers statistically detected and the 271 candidate records thoroughly cataloged and verified, the pipeline advances to **Day 13: Domain Outlier Treatment Decisioning, Sensitivity Analysis & Accounting Invariant Retention Proofs**:
1. **Mathematical Invariant Retention Proofs:** Formulate the algebraic proof showing that Winsorization/capping breaks accounting consistency ($\$24.00 \ne 5 \times \$5.00$) and deletion inflicts $\$6,775.00$ revenue loss.
2. **Sensitivity Analysis:** Model the quantitative impact of capping vs deletion vs retention on storewide KPIs (AOV, median spend, skewness, variance).
3. **Formal Outlier Policy Certification:** Formulate and sign off on the enterprise policy to retain the 271 \$25.00 transactions as certified authentic business boundaries.
