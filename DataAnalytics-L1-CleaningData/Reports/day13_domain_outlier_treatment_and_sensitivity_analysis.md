# ☕ Cafe Sales – Day 13 Report: Domain Outlier Treatment Decisioning, Sensitivity Modeling & Accounting Invariant Retention Proofs
**Oasis Infobyte Data Analytics Internship – Task Level 1**  
**Milestone:** Day 13 of 18 – Domain Outlier Treatment Decisioning, Multi-Scenario Sensitivity Modeling, Algebraic Accounting Invariant Proofs & Certified Enterprise Retention  
**Target Asset:** `Data/dirty_cafe_sales.csv` (10,000 rows × 8 columns)  
**Status:** Certified Production Policy C Sign-Off & 100.00% Accounting Invariant Preservation  

---

## 1. Executive Summary & Milestone Scope

On **Day 13**, the cafe sales data cleaning and transformation pipeline addressed the critical operational and financial governance question established on Day 12: **how should statistical outlier candidates in transactional retail data be treated?**

On Day 12, non-parametric Tukey's Fences ($1.5 \times \text{IQR}$) isolated exactly **271 candidate transactions** ($2.71\%$ of the dataset) with $\text{Total Spent} = \$25.00$ exceeding the upper inner fence boundary of $\$24.00$. All 271 orders represent purchases of **5 units of Salad at the canonical menu price of \$5.00 per unit**, totaling **\$6,775.00 USD in valid gross commercial revenue (7.58% of annual turnover)**. Importantly, zero observations violated the 3-sigma fence ($|Z| \le 2.6722 < 3.0$), and zero extreme anomalies ($> \$36.00$) were present.

Day 13 formulated, modeled, and stress-tested three competing enterprise treatment policies:
1. **Policy A: Naive Deletion / Listwise Dropping** (purging all 271 orders where $\text{Total Spent} > \$24.00$)
2. **Policy B: Winsorization / Fence Capping** (clamping `Total Spent` from $\$25.00$ down to the $\$24.00$ upper fence)
3. **Policy C: Certified Domain-Informed Retention** (retaining all 271 records uncorrupted with verified commercial validity)

Through mathematical proofs, multi-scenario sensitivity modeling, and relational accounting integrity audits, Day 13 formally proves that **Policy C is the only mathematically consistent, financially accurate, and operationally defensible choice**.

```text
+---------------------------------------------------------------------------------------------------------------+
|                                DAY 13 OUTLIER TREATMENT DECISIONING SCORECARD                                 |
+---------------------------------------------------------------------------------------------------------------+
|  Metric Dimension                   Policy A (Deletion)   Policy B (Winsor)     Policy C (Retention - Prod)   |
|  ----------------                   -------------------   -----------------     ---------------------------   |
|  Dataset Record Retention           9,729 (97.29%)        10,000 (100.00%)      10,000 (100.00% Certified)    |
|  Gross Store Revenue                $82,620.00 (-7.58%)   $89,124.00 (-0.30%)   $89,395.00 (100% Protected)   |
|  Direct Financial Loss / Distortion -$6,775.00 USD        -$271.00 USD          $0.00 USD (Zero Deflation)    |
|  Average Order Value (AOV)          $8.4921 (-5.00%)      $8.9124 (-0.30%)      $8.9395 (Baseline Accurate)   |
|  Median Transaction Spend           $7.50 (Deflated)      $8.00 (Preserved)     $8.00 (Preserved)             |
|  Spend Variance (Dispersion)        29.7418 (-17.66%)     35.2772 (-2.34%)      36.1214 (Authentic Spread)    |
|  Fisher-Pearson Skewness (g1)       +0.6623 (Truncated)   +0.7686 (Distorted)   +0.8193 (Natural Right-Skew)  |
|  Excess Kurtosis (g2)               -0.6037 (Flattened)   -0.3308 (Art. Peak)   -0.1540 (Mesokurtic Normal)   |
|  Salad Unit Volume Demand           2,479 units (-35.34%) 3,834 units (100%)    3,834 units (100% Protected)  |
|  Salad Category Revenue             $12,395.00 (-35.34%)  $18,899.00 (-1.41%)   $19,170.00 (100% Protected)   |
|  Relational Invariant Violations    0 (Within Subsample)  271 Rows (FATAL 2.71%)0 Rows (100% Adherence)       |
|  Total Dollar Accounting Error      $0.00                 $271.00 Discrepancy   $0.000000 Exact Identity      |
|  Maximum Discrepancy per Record     $0.00                 $1.00 per Record      $0.000000 Exact Identity      |
|  Universal Cell Completeness        77,832 / 77,832 cells 80,000 / 80,000 cells 80,000 / 80,000 (100.00%)    |
|  Enterprise Governance Verdict      REJECTED              REJECTED              CERTIFIED FOR PRODUCTION      |
+---------------------------------------------------------------------------------------------------------------+
```

### Core Day 13 Engineering Achievements:
1. **Algebraic Invariant Invalidation Proof for Winsorization:** Proved that clamping `Total Spent` to $\$24.00$ generates a fatal relational breach ($5 \times \$5.00 = \$25.00 \ne \$24.00$), injecting **$\$271.00$ in accounting discrepancies across 271 records (2.71% violation rate)**.
2. **Survivorship & Demand Bias Quantification for Deletion:** Proved that deleting candidate outliers purges **$\$6,775.00$ USD in valid annual sales (7.58% revenue loss)**, deflates storewide AOV by **$-5.00\%$** ($\$8.49$ vs $\$8.94$), and destroys **$35.34\%$ of all Salad product volume demand (1,355 units erased)**.
3. **Formal Adoption of Policy C (Certified Retention):** Officially certified Policy C for production deployment, preserving 100.00% of transactions ($10,000 / 10,000$), 100.00% of cell completeness ($80,000 / 80,000$), and 100.00% accounting identity compliance ($\Delta = \$0.000000$).
4. **Multi-Scenario Sensitivity Modeling Matrix:** Constructed an exhaustive 15-metric comparative matrix capturing macroeconomic KPIs, distributional moments, and domain metrics across all three policies.
5. **Two Diagnostic Figures Serialized:** Rendered `day13_treatment_policy_sensitivity_tradeoffs.png` and `day13_retention_policy_and_domain_proofs.png` at 300 DPI.
6. **Automated QA Assertion Sign-Off:** Certified 20 comprehensive structural, mathematical, and domain assertions via `verify_day13_treatment_integrity()`.

---

## 2. Theoretical & Mathematical Foundations: Treatment Paradigms

### 2.1 The Relational Accounting Invariant
In relational retail transactional systems, transactional primitives must strictly satisfy the deterministic accounting invariant $\mathcal{I}$:

$$\mathcal{I}: \quad \text{Total Spent}_i \equiv \text{Quantity}_i \times \text{Price Per Unit}_i \quad \forall i \in \{1, \dots, N\}$$

Where:
- $\text{Quantity}_i \in \{1, 2, 3, 4, 5\} \subset \mathbb{Z}^+$
- $\text{Price Per Unit}_i \in \{\$1.00, \$1.50, \$2.00, \$3.00, \$4.00, \$5.00\} \subset \mathbb{R}^+$
- $\text{Total Spent}_i \in \mathbb{R}^+$

Any data transformation $\mathcal{T}: \mathbf{X} \to \mathbf{X}'$ that alters any of these three attributes without proportional re-estimation of the other two breaks $\mathcal{I}$, creating an irreconcilable discrepancy $\Delta_i$:

$$\Delta_i = \left| \text{Total Spent}'_i - \left( \text{Quantity}'_i \times \text{Price Per Unit}'_i \right) \right|$$

A pipeline is defined as **accounting-sound** if and only if $\sum_{i=1}^N \Delta_i = 0$.

---

### 2.2 Proof of Relational Invalidation Under Policy B (Winsorization)

**Theorem 1 (Winsorization Relational Inconsistency):**  
*Clamping the dependent transaction feature $\text{Total Spent}$ to the Tukey upper fence $F_U = \$24.00$ while preserving the primitive features $\text{Quantity} = 5$ and $\text{Price Per Unit} = \$5.00$ guarantees the violation of the accounting invariant $\mathcal{I}$ on all candidate records, producing a systemic accounting error.*

**Proof:**  
Let the candidate set $\mathcal{C}$ be the subset of transactions exceeding $F_U = \$24.00$:
$$\mathcal{C} = \{ i \in \{1, \dots, N\} \mid \text{Total Spent}_i > \$24.00 \}$$

From Day 12's empirical census, $|\mathcal{C}| = 271$. For every transaction $j \in \mathcal{C}$:
$$\text{Item}_j = \text{'Salad'}, \quad \text{Quantity}_j = 5, \quad \text{Price Per Unit}_j = \$5.00, \quad \text{Total Spent}_j = \$25.00$$

Applying Winsorization capping at $F_U = \$24.00$:
$$\text{Total Spent}'_j = \min(\text{Total Spent}_j, \$24.00) = \$24.00$$

Evaluating invariant $\mathcal{I}$ on transaction $j$:
$$\Delta_j = | \$24.00 - (5 \times \$5.00) | = | \$24.00 - \$25.00 | = \$1.00$$

Since $\Delta_j = \$1.00 > 0$ for all $j \in \mathcal{C}$:
$$\sum_{j \in \mathcal{C}} \Delta_j = 271 \times \$1.00 = \$271.00$$

To resolve this discrepancy, the engineer would be forced to choose between two unacceptable corruptions:
1. **Corrupt Price Per Unit:** Recompute $\text{Price Per Unit}'_j = \frac{\$24.00}{5} = \$4.80$. This violates catalog domain integrity, as no item on the menu sells for $\$4.80$, contradicting Day 7's certified menu price of Salad (\$5.00).
2. **Corrupt Quantity:** Recompute $\text{Quantity}'_j = \frac{\$24.00}{5.00} = 4.8$ units. This violates physical discrete reality, as salads cannot be sold in fractional $0.8$ portions at POS terminals.

**Conclusion:** Policy B introduces irremediable database corruption and is disqualified from enterprise use. $\blacksquare$

---

### 2.3 Proof of Commercial & Survivorship Distortion Under Policy A (Deletion)

**Theorem 2 (Listwise Deletion Survivorship & Demand Distortion):**  
*Purging candidate transactions $\mathcal{C}$ introduces severe truncation bias into financial and inventory forecasting, destroying $7.58\%$ of annual turnover and over $35\%$ of product category demand.*

**Proof:**  
Let total store revenue under Policy C be $R_C = \sum_{i=1}^N \text{Total Spent}_i = \$89,395.00$.  
Under Policy A, the retained index set is $\mathcal{R}_A = \{1, \dots, N\} \setminus \mathcal{C}$, where $|\mathcal{R}_A| = 10,000 - 271 = 9,729$.

The revenue under Policy A is:
$$R_A = \sum_{k \in \mathcal{R}_A} \text{Total Spent}_k = R_C - \sum_{j \in \mathcal{C}} \text{Total Spent}_j = \$89,395.00 - (271 \times \$25.00) = \$82,620.00$$

The percentage revenue loss is:
$$\mathcal{L}_{\text{Rev}} = \frac{\$89,395.00 - \$82,620.00}{\$89,395.00} \times 100\% = \frac{\$6,775.00}{\$89,395.00} \times 100\% = 7.5787\% \approx 7.58\%$$

For the Salad product category specifically, total annual demand in units sold is:
$$D_{\text{Salad, Total}} = \sum_{i: \text{Item}_i = \text{'Salad'}} \text{Quantity}_i = 3,834 \text{ units}$$

The units destroyed by Policy A are:
$$D_{\text{Salad, Lost}} = 271 \times 5 = 1,355 \text{ units}$$

The proportional destruction of category inventory demand is:
$$\mathcal{L}_{\text{Demand}} = \frac{1,355}{3,834} \times 100\% = 35.3417\% \approx 35.34\%$$

**Conclusion:** Deletion artificially erases more than one-third of customer demand for fresh salads, inducing stockouts, misallocating raw ingredient procurement, and underreporting fiscal tax liabilities by $\$6,775.00$. Policy A is disqualified from enterprise use. $\blacksquare$

---

## 3. Multi-Scenario Sensitivity Modeling Matrix

The complete quantitative sensitivity simulation across the three candidate policies is detailed below:

| Metric Dimension | Baseline / Policy C (Retention) | Policy B (Winsorization) | Policy A (Naive Deletion) | Sensitivity Shift (C $\to$ A) | Operational Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Transaction Records ($N$)** | **10,000** | 10,000 | 9,729 | **-271 (-2.71%)** | Loss of top-decile ticket orders |
| **Retention Rate (%)** | **100.00%** | 100.00% | 97.29% | -2.71% | Unacceptable destructive purging |
| **Gross Store Revenue ($)** | **\$89,395.00** | \$89,124.00 | \$82,620.00 | **-\$6,775.00 (-7.58%)** | Severe underreporting of enterprise turnover |
| **Revenue Delta ($)** | **\$0.00** | -\$271.00 | -\$6,775.00 | -\$6,775.00 | Policy B deflates; Policy A erases |
| **Average Order Value (AOV)** | **\$8.9395** | \$8.9124 | \$8.4921 | **-\$0.4474 (-5.00%)** | Depresses executive basket-size metric |
| **Median Spend ($)** | **\$8.0000** | \$8.0000 | \$7.5000 | -\$0.5000 (-6.25%) | Shifts sample median downward |
| **Standard Deviation ($\sigma$)** | **\$6.0101** | \$5.9395 | \$5.4536 | -\$0.5565 (-9.26%) | Compresses natural commercial spread |
| **Variance ($\sigma^2$)** | **36.1214** | 35.2772 | 29.7418 | **-6.3796 (-17.66%)** | Severe suppression of stochastic variance |
| **Interquartile Range (IQR)** | **\$8.0000** | \$8.0000 | \$8.0000 | \$0.0000 (0.00%) | Quartile boundary invariance |
| **Skewness ($g_1$)** | **+0.8193** | +0.7686 | +0.6623 | **-0.1570 (-19.16%)** | Artificially curtails natural right-tail |
| **Excess Kurtosis ($g_2$)** | **-0.1540** | -0.3308 | -0.6037 | **-0.4497 (+292% plat)**| Flattens tail into unnatural box distribution |
| **Salad Units Sold** | **3,834 units** | 3,834 units | 2,479 units | **-1,355 units (-35.34%)**| Cripples supply chain demand forecasting |
| **Salad Revenue ($)** | **\$19,170.00** | \$18,899.00 | \$12,395.00 | **-\$6,775.00 (-35.34%)**| Erases over 1/3 of category revenue |
| **Invariant Violations** | **0 (0.00%)** | **271 (2.71%)** | 0 (in subset) | 0 vs 271 | Policy B causes systemic relational breach |
| **Total Discrepancy ($)** | **\$0.000000** | **\$271.00** | \$0.00 | +\$271.00 in B | Policy B creates unreconciled balance error |

---

## 4. Mathematical Accounting Invariant Discrepancy Breakdown

The table below contrasts the mathematical accounting fidelity of each policy:

| Policy Name | Targeted Cohort | Mathematical Formula | Invariant Status | Violation Count | Violation Rate (%) | Max Discrepancy | Total Discrepancy | Systemic Accounting Risk |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Policy A: Naive Deletion** | 271 orders | $\text{Total} = \text{Qty} \times \text{PPU}$ | Adhered in subset | 0 | 0.00% | \$0.00 | \$0.00 | High Survivorship Bias & Data Loss |
| **Policy B: Winsorization** | 271 orders | $\text{Total} = \text{Qty} \times \text{PPU}$ | **VIOLATED** | **271** | **2.71%** | **\$1.00** | **\$271.00** | **Critical Relational & Ledger Failure** |
| **Policy C: Domain Retention**| 271 orders | $\text{Total} = \text{Qty} \times \text{PPU}$ | **CERTIFIED** | **0** | **0.00%** | **\$0.000000** | **\$0.000000** | **Zero Risk (100% Exact Identity)** |

---

## 5. Commercial & Inventory Demand Distortion Analysis

### 5.1 The Danger of Supply Chain De-synchronization
In commercial cafe operations, salads require fresh produce procurement (lettuce, cucumbers, tomatoes, vinaigrette) with short shelf lives (typically 48–72 hours). 

- Under **Policy C (Retention)**, procurement algorithms observe accurate historical demand: **3,834 salads sold annually**, averaging **319.5 salads per month** (approx. 10.5 salads/day).
- Under **Policy A (Deletion)**, historical demand is reported as only **2,479 salads sold annually**, averaging **206.6 salads per month** (approx. 6.8 salads/day).
- **Operational Impact:** Adopting Policy A would lead automated ERP inventory systems to under-order fresh ingredients by **35.34%**, causing chronic midday stockouts, customer dissatisfaction, and revenue abandonment.

### 5.2 Year-Round Temporal Uniformity Proof
Auditing the monthly occurrence of the 271 candidate transactions confirmed that they are not temporal spikes or promotional anomalies:
- **Mean Monthly Orders:** 22.58 orders/month ($\sigma = 5.2$ orders)
- **Minimum Month:** February (13 orders, \$325.00)
- **Maximum Month:** July (32 orders, \$800.00)
- **Consistency:** Transactions occurred in all 12 calendar months of 2023 without interruption.

---

## 6. Enterprise Governance Decision Scorecard

Each treatment policy was evaluated across 6 critical enterprise dimensions weighted by business risk:

| Evaluation Dimension | Weight | Policy A (Naive Deletion) | Policy B (Winsorization) | Policy C (Domain Retention) | Governance Sign-Off |
| :--- | :---: | :--- | :--- | :--- | :---: |
| **1. Dataset Completeness & Retention** | 25% | **FAIL:** Purges 271 valid transactions (2.71% loss) | **PASS:** Preserves 10,000 / 10,000 records | **PASS:** Certified 100.0% retention (10,000 rows) | **Policy C** |
| **2. Revenue Accounting Integrity** | 25% | **FAIL:** Erases \$6,775.00 in valid revenue (-7.58%) | **FAIL:** Artificially deflates revenue by \$271.00 | **PASS:** 100.0% gross revenue protected (\$89,395.00)| **Policy C** |
| **3. Mathematical Invariant Adherence** | 20% | **CONDITIONAL:** Intact on retained subset only | **FATAL FAULT:** 271 accounting breaches ($5 \times \$5 \ne \$24$)| **PASS:** 0 violations across 10,000 rows ($\Delta = \$0$) | **Policy C** |
| **4. Inventory Forecasting Fidelity** | 15% | **FAIL:** Destroys 35.34% of Salad unit volume | **PASS:** Preserves item demand quantities | **PASS:** 100.0% category demand intact (3,834 units)| **Policy C** |
| **5. Bias & Tail Truncation Avoidance** | 10% | **FAIL:** Severe survivorship truncation bias | **FAIL:** Creates artificial point mass spike at \$24 | **PASS:** Preserves natural bounded right-skewed tail| **Policy C** |
| **6. Production Deployment Certification**| 100% | **REJECTED:** Incompatible with financial reporting | **REJECTED:** Violates relational database schema | **CERTIFIED FOR PRODUCTION DEPLOYMENT** | **Policy C** |

---

## 7. Automated Quality Assurance & Assertion Verification Matrix

The pipeline's automated verification engine `verify_day13_treatment_integrity()` executed 20 rigorous assertions certifying the structural, relational, and domain validity of Day 13:

| # | Verification Assertion Condition | Expected Standard | Observed Value | QA Status |
| :---: | :--- | :---: | :---: | :---: |
| **c1** | Record count preservation | Exactly 10,000 rows | 10,000 rows | **PASSED (✅)** |
| **c2** | Canonical schema feature preservation | Exactly 8 columns | 8 columns | **PASSED (✅)** |
| **c3** | Universal cell completeness | 80,000 / 80,000 cells | 80,000 non-null cells | **PASSED (✅)** |
| **c4** | Policy C total gross revenue intact | Exactly \$89,395.00 | \$89,395.00 USD | **PASSED (✅)** |
| **c5** | Policy C zero accounting violations | Max $\Delta < 10^{-4}$ | $\Delta = \$0.000000$ | **PASSED (✅)** |
| **c6** | Policy A deletion record count | Exactly 9,729 records | 9,729 records | **PASSED (✅)** |
| **c7** | Policy A revenue loss quantification | Exactly \$6,775.00 USD | \$6,775.00 USD | **PASSED (✅)** |
| **c8** | Policy A Salad demand destruction | Exactly 35.34% (1,355 units)| 35.34% (1,355 units) | **PASSED (✅)** |
| **c9** | Policy B invariant violation count | Exactly 271 violations | 271 violations | **PASSED (✅)** |
| **c10**| Policy B total dollar discrepancy | Exactly \$271.00 USD | \$271.00 USD | **PASSED (✅)** |
| **c11**| Policy B deflated revenue value | Exactly \$89,124.00 USD | \$89,124.00 USD | **PASSED (✅)** |
| **c12**| Candidate cohort invariant ($5 \times \$5 = \$25$) | 100% adherence on 271 rows | 100% adherence (271/271) | **PASSED (✅)** |
| **c13**| Policy C AOV fidelity | Exactly \$8.9395 USD | \$8.9395 USD | **PASSED (✅)** |
| **c14**| Policy C variance fidelity | Exactly 36.1214 $\$^2$ | 36.1214 $\$^2$ | **PASSED (✅)** |
| **c15**| Policy C skewness fidelity | Exactly +0.8193 | +0.8193 | **PASSED (✅)** |
| **c16**| Policy C excess kurtosis fidelity | Exactly -0.1540 | -0.1540 | **PASSED (✅)** |
| **c17**| Sensitivity summary matrix coverage | 3 evaluated policies | 3 policies evaluated | **PASSED (✅)** |
| **c18**| Accounting discrepancy contrast | Policy C: 0 vs Policy B: 271 | 0 vs 271 violations | **PASSED (✅)** |
| **c19**| Governance scorecard verdict | Policy C Certified | Certified for Production | **PASSED (✅)** |
| **c20**| Primary key continuity | 10,000 matching TXN IDs | 100% ID Match | **PASSED (✅)** |

---

## 8. Diagnostic Visual Artifacts

Two publication-grade diagnostic visual figures (300 DPI) were generated and serialized to `Reports/figures/`:

1. **`Reports/figures/day13_treatment_policy_sensitivity_tradeoffs.png`**  
   *A quad layout illustrating:*
   - **Panel 1: Multi-Scenario Financial Revenue & Salad Impact:** Bar chart contrasting total store gross revenue (\$82,620 vs \$89,124 vs \$89,395) and Salad category revenue across Policies A, B, and C.
   - **Panel 2: Moment Sensitivity & Statistical Dispersion Drift:** Contrasts Average Order Value (\$8.49 vs \$8.91 vs \$8.94), Variance (29.74 vs 35.28 vs 36.12), and Skewness (+0.66 vs +0.77 vs +0.82) across policies.
   - **Panel 3: Accounting Invariant Violations & Dollar Error:** Displays the catastrophic 271 violation count and \$271.00 ledger error created by Policy B Winsorization versus \$0.00 under Policy C.
   - **Panel 4: Enterprise Governance Policy Decision Scorecard:** Formatted scorecard matrix highlighting pass/fail status across the 6 core operational dimensions.

2. **`Reports/figures/day13_retention_policy_and_domain_proofs.png`**  
   *A quad layout illustrating:*
   - **Panel 1: Geometric Accounting Invariant & Capping Discrepancy Proof:** Scatter plot showing Salad transactions along the linear ray $\text{Total} = \text{Quantity} \times \$5.00$, illustrating how Policy C retains the valid point $(5, \$25.00)$ while Policy B forces an illegal off-ray point $(5, \$24.00)$.
   - **Panel 2: Empirical Cumulative Distribution Function (ECDF) Tail Comparison:** Overlays ECDFs for Policies A, B, and C, exposing Policy A's sharp truncation and Policy B's artificial point mass spike at \$24.00.
   - **Panel 3: Monthly Commercial Protection of \$25.00 Orders Across 2023:** Stacked bar chart showing the steady year-round contribution of candidate orders (~$564.58/month) across all 12 calendar months.
   - **Panel 4: Salad Category Inventory Demand Protection:** Bar chart comparing Salad unit demand and revenue between Policy A (2,479 units) and Policy C (3,834 units), highlighting the 35.34% demand deficit prevented by Policy C.

---

## 9. Business, Macroeconomic & Operations Impact Analysis

### 9.1 Fiscal Compliance & Revenue Defense
Retaining the 271 candidate transactions defends **\$6,775.00 USD in legitimate commercial turnover**. Under GAAP and IFRS revenue recognition standards, deleting valid cash and digital transactions constitutes material financial misstatement. Policy C guarantees 100% tax and financial audit compliance.

### 9.2 Customer Lifetime Value & VIP Basket Protection
Customers ordering 5 premium salads represent high-value corporate lunch orders or catering tickets. Naively flagging these transactions as "anomalies" and dropping them would penalize high-ticket customer cohorts in downstream Customer Lifetime Value (CLV) and market-basket association models.

---

## 10. Master Entity, Completeness & Quality Lifecycle Matrix (Days 1–13)

The master table below summarizes the cumulative progression of dataset health, cleanliness, and structural integrity across the entire project lifecycle through Day 13:

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
| **Day 12: Outliers**| Statistical dual-modeling| 10,000 | **80,000** | **100.00%** | **0** | **100% Unique** | **0 Violations** | 271 Mild / 0 Extreme | \$89,395.00 |
| **Day 13: Treatment**| Multi-scenario retention| **10,000** | **80,000** | **100.00%** | **0** | **100% Unique** | **0 Violations** | **Policy C Certified** | **\$89,395.00 (100%)**|

---

## 11. Strategic Handoff to Days 14–16

With outlier treatment formally certified and all 10,000 transactions proven mathematically and commercially authentic, the pipeline advances to **Days 14–16: Master Before vs. After Comparative Audit and Clean Dataset Export**:
1. **Master Comparative Audit:** Compile side-by-side distributions, missingness profiles, and data quality indicators contrasting the raw `dirty_cafe_sales.csv` against the cleaned production DataFrame.
2. **Clean Dataset Serialization:** Export the production-certified dataset to `Data/clean_cafe_sales.csv` with optimized schema dtypes and verified primary key uniqueness.
3. **Data Dictionary Generation:** Serialize a comprehensive technical data dictionary detailing canonical column definitions, permissible categorical levels, and data type specifications.
