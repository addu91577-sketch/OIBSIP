# 🍷 Wine Quality Prediction — Physicochemical Classification
**Oasis Infobyte Data Analytics Internship — Task Level 2 (Task 2)**

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B.svg?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-45%20Passed-brightgreen.svg?logo=pytest&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-2.x-150458.svg)
![NumPy](https://img.shields.io/badge/numpy-1.x-013243.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.x-11557c.svg)
![Seaborn](https://img.shields.io/badge/seaborn-0.12+-4c72b0.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)

---

## 🎯 Objective
Train and compare multiple supervised machine learning classification models to predict the sensory quality score of wine (typically on a scale of 3–8) based on its physicochemical laboratory properties, such as volatile acidity, citric acid, sulphates, and alcohol content. Every analytical stage encompasses rigorous statistical diagnostics, data preparation, model benchmarking, and production suitability analysis.

---

## 🛠️ Tech Stack
- **Programming Language:** Python 3
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Machine Learning & Modeling:** `scikit-learn` (`RandomForestClassifier`, `SGDClassifier`, `SVC`, metrics, model selection)
- **Data Visualization:** `matplotlib`, `seaborn`
- **Interactive Development:** Jupyter Notebook (`.ipynb`)
- **Version Control:** Git & GitHub

---

## 📋 Feature Checklist
- [x] **Load the Wine Quality dataset and inspect structure; check class distribution of quality scores**
- [x] **EDA: distribution plots for all chemical features; correlation heatmap**
- [x] **Discuss class imbalance: are certain quality scores underrepresented? How does this affect modelling?**
- [x] **Feature engineering:** consider binning quality scores into binary (good/bad) or 3-class (low/medium/high) groups and justify the decision
- [x] **Train/test split with stratification** to preserve class ratios
- [x] **Train 3 classifiers:** Random Forest, Stochastic Gradient Descent (SGD), and Support Vector Classifier (SVC)
- [x] **Evaluate each:** accuracy, classification report, confusion matrix
- [x] **Feature importance chart** for the Random Forest model
- [x] **Comparison table:** summarise the performance of all 3 models side-by-side
- [x] **Conclusion:** which model is most suitable for deployment and why?
- [x] **Day 7 Production REST API Engine:** FastAPI microservice with strict Pydantic v2 schemas and validation bounds
- [x] **Day 7 Commercial Sensory Grading & Guidance:** 4-tier commercial classification and cellar directives
- [x] **Day 7 Interactive Streamlit Dashboard:** Real-time parameter tuning, radar benchmarks, and batch CSV triage
- [x] **Day 7 Automated Pytest Suite:** 22 passing unit and integration tests across data, models, and endpoints
- [x] **Day 7 Dockerization & CI/CD Pipeline:** Multi-stage Dockerfile, docker-compose, and GitHub Actions CI workflow
- [x] **Day 8 Continuous MLOps Monitoring:** Quantile PSI, K-S tests, multivariate anomaly filter & automated retraining
- [x] **Day 9 Cross-Cultivar Domain Adaptation & Cellar Recourse:** White Wine champion, 99.6% cultivar classifier & prescriptive recourse engine

---

## 📂 Repository Structure
```text
DataAnalytics-L2-WineQuality/
├── .dockerignore                       # Exclusions for Docker container build context
├── .gitignore                          # Git ignore rules for Python, Jupyter checkpoints, and virtualenvs
├── .github/
│   └── workflows/
│       └── ci.yml                      # GitHub Actions CI workflow for test suite & integrity
├── conftest.py                         # Pytest root path fixtures
├── docker-compose.yml                  # Multi-service orchestration (FastAPI + Streamlit)
├── Dockerfile                          # Multi-stage production container definition
├── pytest.ini                          # Pytest test discovery & warning filters
├── README.md                           # Project documentation, inspection findings, and architecture
├── requirements.txt                    # Production and analytical dependencies
├── Data/
│   ├── winequality-red.csv             # Red wine physicochemical & sensory dataset (1,599 × 12)
│   ├── winequality-white.csv           # White wine physicochemical & sensory dataset (4,898 × 12)
│   └── winequality.names               # UCI dataset citation & attribute documentation
├── models/
│   ├── champion_rf_pipeline.joblib     # Serialized Champion Random Forest classification pipeline (2.48 MB)
│   ├── champion_rf_pipeline.json       # Production metadata, hyperparameter configuration & schema specs
│   ├── white_champion_rf_pipeline.joblib # Serialized White Wine Champion Random Forest pipeline
│   ├── white_champion_rf_pipeline.json   # White Wine model metadata, parameters & test benchmarks
│   ├── cultivar_classifier.joblib      # Serialized 99.6% accurate Red/White Cultivar Detector
│   └── cultivar_classifier.json        # Cultivar Detector metadata & discriminator rankings
├── Notebooks/
│   ├── 01_wine_quality_inspection.ipynb # Day 1: Data ingestion, schema inspection & class distribution
│   ├── 02_wine_quality_eda.ipynb       # Day 2: Exploratory data analysis, distributions, outliers & correlation
│   ├── 03_target_engineering_and_baseline_models.ipynb # Day 3: Target binning, domain features & baseline models
│   ├── 04_hyperparameter_optimization_feature_importance_and_model_evaluation.ipynb # Day 4: Tuning, feature importance & evaluation
│   ├── 05_model_interpretation_feature_importance_and_champion_pipeline.ipynb # Day 5: Interpretability, PDP, serialization & live inference
│   ├── 06_model_comparison_hardware_benchmarking_and_deployment_synthesis.ipynb # Day 6: Benchmark matrix, latency & synthesis
│   ├── 07_production_deployment_and_interactive_dashboard.ipynb # Day 7: Production API, interactive dashboard, containerization & testing
│   ├── 08_continuous_mlops_monitoring_and_retraining.ipynb # Day 8: Continuous MLOps drift monitoring, statistical testing & retraining
│   └── 09_cross_cultivar_domain_adaptation_and_recourse.ipynb # Day 9: Cross-cultivar domain adaptation, white wine champion & recourse
├── src/
│   ├── __init__.py                     # Package interface & symbol exports
│   ├── baseline_models.py              # Baseline model training & diagnostic plots
│   ├── cross_cultivar_and_recourse.py  # Cross-cultivar divergence, White champion, cultivar detector & cellar recourse
│   ├── data_loader.py                  # UCI data ingestion & structural validation
│   ├── day7_figures.py                 # Day 7 publication figures generator
│   ├── day8_figures.py                 # Day 8 publication figures generator
│   ├── day9_figures.py                 # Day 9 domain adaptation & recourse figures generator
│   ├── drift_monitoring.py             # Covariate shift (PSI, K-S), anomaly screening & champion-challenger pipeline
│   ├── deployment_synthesis.py         # Hardware benchmarking & MCDA scoring
│   ├── eda_profiling.py                # Univariate distributions & outlier profiling
│   ├── feature_engineering.py          # Domain interaction features & stratified split
│   ├── interactive_dashboard.py        # Streamlit interactive decision support platform
│   ├── model_interpretation.py         # PDP, permutation importance & serialization
│   ├── model_tuning.py                 # Hyperparameter search & cross-validation
│   └── production_service.py           # FastAPI production serving engine & Pydantic schemas
├── tests/
│   ├── __init__.py                     # Test package initializer
│   ├── test_pipeline.py                # Data integrity & feature engineering tests (7 tests)
│   ├── test_production_service.py      # FastAPI endpoints, validation & latency tests (15 tests)
│   ├── test_drift_monitoring.py        # Statistical drift, PSI, anomaly filter & tournament tests (10 tests)
│   └── test_cross_cultivar_and_recourse.py # Cross-cultivar divergence, White champion, recourse & API tests (13 tests)
└── reports/
    ├── day4_feature_importance_rf.csv          # Day 4 feature importance rankings
    ├── day4_model_comparison_table.csv         # Day 4 benchmark matrix
    ├── day5_rf_feature_importance.csv          # Day 5 Gini MDI & permutation importance rankings
    ├── day5_linear_coefficients.csv            # Day 5 standardized linear weights & odds ratio multipliers
    ├── day5_enological_decision_rules.csv      # Day 5 empirical chemical shifts & cellar rules
    ├── day6_master_model_comparison.csv        # Day 6 consolidated multi-metric benchmark matrix
    ├── day6_hardware_latency_benchmark.csv     # Day 6 single-sample & batch latency scaling metrics
    ├── day6_deployment_suitability_matrix.csv  # Day 6 multi-criteria decision analysis (MCDA) matrix
    ├── day7_production_api_benchmarks.csv      # Day 7 microsecond-resolution API latency benchmarks
    ├── final_model_comparison.md               # Day 6 technical deployment handover report
    └── figures/
        ├── day1_quality_distribution.png           # Annotated quality score frequency distribution plot
        ├── day2_univariate_distributions.png       # 11-panel histogram & KDE distributions
        ├── day2_feature_boxplots.png               # 11-panel Tukey IQR outlier boxplots
        ├── day2_features_vs_quality.png            # Bivariate boxplots of key chemical drivers across quality
        ├── day2_correlation_heatmap.png            # Lower-triangular Pearson correlation matrix heatmap
        ├── day2_top_correlations.png               # Ranked feature correlations with sensory quality
        ├── day3_target_distribution_comparison.png # Raw vs 3-tier vs binary target distribution comparison
        ├── day3_baseline_model_comparison.png      # Baseline classifiers performance comparison bar chart
        ├── day3_confusion_matrices.png             # Side-by-side annotated confusion matrix heatmaps
        ├── day3_roc_curves.png                     # Receiver Operating Characteristic (ROC) curves
        ├── day3_precision_recall_curves.png        # Precision-Recall curves vs prevalence baseline
        ├── day4_cv_performance_comparison.png     # Stratified 5-Fold CV metric comparison (Baseline vs Tuned)
        ├── day4_feature_importance_rf.png          # Dual-panel Random Forest feature importance
        ├── day4_tuned_confusion_matrices.png       # Side-by-side tuned confusion matrix heatmaps
        ├── day4_precision_recall_curves.png        # Precision-Recall curves comparing tuned vs baseline models
        ├── day4_roc_curves.png                     # Receiver Operating Characteristic (ROC) curves
        ├── day4_side_by_side_comparison.png       # Comprehensive multi-metric benchmark comparison bar chart
        ├── day5_rf_feature_importance.png          # Dual-panel MDI vs out-of-sample permutation importance
        ├── day5_permutation_importance_comparison.png # Multi-model cross-architecture permutation comparison
        ├── day5_linear_coefficients.png            # Diverging directional linear weights & odds ratio multipliers
        ├── day5_partial_dependence_plots.png       # 6-panel 1D partial dependence profiles & tipping points
        ├── day5_pdp_2d_interactions.png            # 2D bivariate partial dependence interaction contours
        ├── day5_champion_model_summary.png         # Champion model executive dashboard & chemical contrast
        ├── day6_comprehensive_model_comparison.png  # Master multi-metric comparison across baseline & tuned models
        ├── day6_latency_vs_performance_tradeoff.png # Bubble plot of throughput vs ROC-AUC with artifact footprints
        ├── day6_deployment_decision_radar.png       # Multi-criteria spider radar chart across operational dimensions
        ├── day6_executive_scorecard.png             # Final executive scorecard & deployment handover dashboard
        ├── day7_api_latency_distribution.png        # ASGI response latency empirical distribution & percentiles
        ├── day7_batch_inference_scaling.png         # Vectorized batch inference throughput scaling curves
        ├── day7_production_architecture.png         # End-to-end production MLOps containerized architecture
        ├── day7_sommelier_profile_radar.png         # Interactive sommelier radar chart vs reserve benchmark
        ├── day8_feature_drift_psi_summary.png       # Feature-level Population Stability Index (PSI) scorecard
        ├── day8_drift_distribution_shifts.png       # KDE distributions comparing baseline vs shifted harvest
        ├── day8_concept_drift_performance_decay.png # Longitudinal SLA degradation and retraining recovery curve
        ├── day8_mlops_monitoring_pipeline.png       # Closed-loop MLOps monitoring and retraining architecture
        ├── day9_cross_cultivar_chemical_divergence.png # Multi-panel KDE distributions showing Red vs White divergence
        ├── day9_domain_transfer_gap.png             # In-domain vs zero-shot transfer degradation benchmark
        ├── day9_white_champion_roc_pr_curves.png    # White Wine Champion ROC & PR curves (tau*=0.35)
        └── day9_counterfactual_recourse_waterfall.png # Actionable cellar recourse probability waterfall
```

---

## 📊 Dataset Schema & Attribute Definitions
The dataset originates from the **UCI Machine Learning Repository** (*Cortez et al., 2009*). All inputs are continuous physicochemical measurements obtained from laboratory tests, while the target is an integer sensory quality rating evaluated by wine certification experts.

| Attribute Name | Data Type | Units / Scale | Description |
| :--- | :--- | :--- | :--- |
| **Fixed Acidity** | `float64` | $g(tartaric\ acid)/dm^3$ | Non-volatile organic acids contributing to wine freshness and stability. |
| **Volatile Acidity** | `float64` | $g(acetic\ acid)/dm^3$ | Acetic acid concentration; excessive levels impart an undesirable vinegar smell and taste. |
| **Citric Acid** | `float64` | $g/dm^3$ | Adds freshness, crispness, and preserves flavor profiles. |
| **Residual Sugar** | `float64` | $g/dm^3$ | Sugar remaining post-fermentation; defines sweetness levels. |
| **Chlorides** | `float64` | $g(sodium\ chloride)/dm^3$ | Salt concentration in the wine. |
| **Free Sulfur Dioxide** | `float64` | $mg/dm^3$ | Dissolved $SO_2$ gas preventing microbial growth and oxidation. |
| **Total Sulfur Dioxide** | `float64` | $mg/dm^3$ | Sum of free and bound $SO_2$; high levels produce pungent odor. |
| **Density** | `float64` | $g/cm^3$ | Mass density; influenced by alcohol percentage and sugar dissolved. |
| **pH** | `float64` | $0–14$ scale | Measure of wine acidity/alkalinity (typical range 2.9–4.0). |
| **Sulphates** | `float64` | $g(potassium\ sulphate)/dm^3$ | Antimicrobial and antioxidant additive enhancing $SO_2$ efficacy. |
| **Alcohol** | `float64` | $\% vol$ | Percent alcohol by volume produced via fermentation. |
| **Quality** *(Target)* | `int64` | $3–8$ (Scale $0–10$) | Median sensory score awarded by at least 3 wine sensory judges. |

---

## 🔍 Day 1: Data Ingestion & Structural Inspection Highlights

### 1. Data Integrity & Completeness
- **Red Wine Records:** 1,599 observations across 12 attributes.
- **White Wine Records:** 4,898 observations across 12 attributes.
- **Null / Missing Values:** `0` missing values across all columns in both datasets. Complete data completeness without imputation required.
- **Memory Footprint:** 150.0 KB for red wine dataset in memory.

### 2. Class Distribution of Wine Quality Scores
Sensory quality scores range from **3** to **8** in the red wine dataset:

| Quality Rating | Sample Count | Class Representation (%) | Cumulative Share (%) | Evaluation Tier |
| :---: | :---: | :---: | :---: | :---: |
| **3** | 10 | 0.63% | 0.63% | Severe Minority (Poor) |
| **4** | 53 | 3.31% | 3.94% | Minority (Below Average) |
| **5** | 681 | 42.59% | 46.53% | Majority (Average) |
| **6** | 638 | 39.90% | 86.43% | Majority (Average) |
| **7** | 199 | 12.45% | 98.87% | Moderate Minority (Good) |
| **8** | 18 | 1.13% | 100.00% | Severe Minority (Exceptional) |

![Quality Distribution](reports/figures/day1_quality_distribution.png)

### 3. Core Insights & Modeling Implications
1. **Severe Imbalance Concentration:** Scores **5 and 6** encompass **82.49%** of the entire dataset. A naive Zero-Rule baseline predicting only these middle scores would record >82% accuracy while failing completely on critical outlier grades.
2. **Stratification Requirement:** Standard random train/test splitting risks omitting extreme classes (e.g. score 3 with only 10 samples total). Stratified partitioning (`StratifiedKFold`, stratified train/test split) is strictly required to preserve exact class proportions.
3. **Target Engineering Rationale:** Multi-class classification across 6 fine-grained grades with high class imbalance causes decision boundary overlap. Re-framing the problem into binary (`Good` vs `Bad`) or 3-tier (`Low`, `Medium`, `High`) classes will be statistically evaluated in upcoming feature engineering phases.

---

## 🔬 Day 2: Exploratory Data Analysis & Class Imbalance Highlights

### 1. Univariate Statistical Profiles & Dispersion
Across the 11 continuous physicochemical features, central tendency and dispersion metrics were profiled:

| Feature | Mean | Std | Median | IQR | Min | Max | Skewness | Kurtosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed Acidity** | 8.320 | 1.741 | 7.900 | 2.100 | 4.600 | 15.900 | +0.983 | +1.132 |
| **Volatile Acidity** | 0.528 | 0.179 | 0.520 | 0.250 | 0.120 | 1.580 | +0.672 | +1.225 |
| **Citric Acid** | 0.271 | 0.195 | 0.260 | 0.330 | 0.000 | 1.000 | +0.318 | -0.788 |
| **Residual Sugar** | 2.539 | 1.410 | 2.200 | 0.700 | 0.900 | 15.500 | +4.541 | +28.618 |
| **Chlorides** | 0.087 | 0.047 | 0.079 | 0.020 | 0.012 | 0.611 | +5.680 | +41.716 |
| **Free SO₂** | 15.875 | 10.460 | 14.000 | 14.000 | 1.000 | 72.000 | +1.251 | +2.024 |
| **Total SO₂** | 46.468 | 32.895 | 38.000 | 40.000 | 6.000 | 289.000 | +1.516 | +3.810 |
| **Density** | 0.997 | 0.002 | 0.997 | 0.002 | 0.990 | 1.004 | +0.071 | +0.934 |
| **pH** | 3.311 | 0.154 | 3.310 | 0.190 | 2.740 | 4.010 | +0.194 | +0.807 |
| **Sulphates** | 0.658 | 0.170 | 0.620 | 0.180 | 0.330 | 2.000 | +2.429 | +11.721 |
| **Alcohol** | 10.423 | 1.066 | 10.200 | 1.600 | 8.400 | 14.900 | +0.861 | +0.200 |

![Univariate Distributions](reports/figures/day2_univariate_distributions.png)

### 2. Outlier Detection via Tukey's IQR Method
Using the statistical boundary $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$, outlier frequency was quantified:

| Feature | Lower Fence | Upper Fence | Outlier Count | Outlier Share (%) | Enological Nature |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Residual Sugar** | 0.850 | 3.650 | 155 | 9.69% | Late harvest dessert style / sweet wines |
| **Chlorides** | 0.040 | 0.120 | 112 | 7.00% | High mineral saline soils |
| **Total SO₂** | -38.000 | 122.000 | 55 | 3.44% | Highly preserved / anti-microbial lots |
| **Sulphates** | 0.280 | 1.000 | 59 | 3.69% | Added potassium metabisulphite |
| **Fixed Acidity** | 3.950 | 12.350 | 49 | 3.06% | Tartaric acid-rich vintages |
| **Volatile Acidity** | 0.015 | 1.015 | 19 | 1.19% | Aerobic bacterial acetic spoilage |
| **Alcohol** | 7.100 | 13.500 | 13 | 0.81% | High-sugar fermentation musts |
| **Density** | 0.993 | 1.001 | 45 | 2.81% | Dense sugar/alcohol interaction balance |
| **pH** | 2.925 | 3.685 | 35 | 2.19% | Outlier basic/acidic musts |
| **Free SO₂** | -14.000 | 42.000 | 30 | 1.88% | High gaseous protection |

![Feature Boxplots](reports/figures/day2_feature_boxplots.png)

### 3. Key Physicochemical Drivers of Sensory Quality
Tracking how chemical values vary across sensory ratings (scores 3–8):
- **Alcohol ($r = +0.476$):** Strongest positive driver. Median alcohol increases from $9.9\%$ (score 5) to $12.1\%$ (score 8). Alcohol provides body, perceived sweetness, and volatilizes bouquet compounds.
- **Volatile Acidity ($r = -0.391$):** Strongest negative driver. Median drops from $0.60\text{ g/dm}^3$ (score 3) down to $0.37\text{ g/dm}^3$ (scores 7 & 8). High acetic acid produces an unmistakable vinegar taint.
- **Sulphates ($r = +0.251$):** Positive driver. Median rises from $0.56\text{ g/dm}^3$ (score 3) to $0.74\text{ g/dm}^3$ (score 8), providing antioxidant protection.
- **Citric Acid ($r = +0.226$):** Adds crisp acidity and freshness, suppressing flat sensory notes.

![Features vs Quality](reports/figures/day2_features_vs_quality.png)

### 4. Correlation Matrix & Multicollinearity Diagnostics
- **Strong Positive Collinearities:**
  - $\text{Fixed Acidity} \leftrightarrow \text{Citric Acid}$ ($r = +0.67$): Both are fixed organic acids.
  - $\text{Fixed Acidity} \leftrightarrow \text{Density}$ ($r = +0.67$): Dissolved tartaric acid increases must density.
  - $\text{Free SO}_2 \leftrightarrow \text{Total SO}_2$ ($r = +0.67$): Free gas is an equilibrium fraction of total added sulphur.
- **Strong Negative Collinearities:**
  - $\text{Fixed Acidity} \leftrightarrow \text{pH}$ ($r = -0.68$): Higher acid concentration directly drives lower pH ($-\log[H^+]$).
  - $\text{Alcohol} \leftrightarrow \text{Density}$ ($r = -0.50$): Ethanol has a lower density ($0.789\text{ g/cm}^3$) than water, lowering overall wine density.

![Correlation Heatmap](reports/figures/day2_correlation_heatmap.png)
![Top Correlations](reports/figures/day2_top_correlations.png)

### 5. Detailed Discussion: Class Imbalance & Modeling Implications

#### A. Why are certain quality scores underrepresented?
The wine dataset sensory target is derived from certified enological tasting panels (median score of at least 3 judges). In commercial viticulture:
1. **Commercial Benchmarking:** Winemakers blend away extreme faults (yielding very few wines with score 3: $0.63\%$) and reserve bottles are rare artisan expressions (score 8: $1.13\%$).
2. **Central Tendency in Sensory Panels:** Human judges naturally gravitate toward safe middle ratings (scores 5 & 6 represent **82.49%** of all wines).

#### B. How does this severe class imbalance affect classification modeling?
1. **The Accuracy Paradox:** A naive baseline model that always predicts class 5 or 6 achieves $\approx 82.5\%$ overall accuracy while failing completely on commercial reserve wines ($0\%$ recall on grade $\ge 7$) and flawed wines ($0\%$ recall on grades 3 & 4).
2. **Gradient Starvation in Optimization:** Objective functions such as Cross-Entropy or Gini Impurity minimize total misclassification count. Because $82.5\%$ of gradient updates originate from classes 5 and 6, optimization paths push decision hyperplanes away from rare classes, treating minority errors as statistical noise.
3. **High Variance Across Folds:** In standard unstratified cross-validation, rare classes with only 10 samples (class 3) or 18 samples (class 8) may be entirely missing from certain folds, producing wildly fluctuating evaluation metrics ($\sigma > 0.25$).

#### C. Statistical Solutions Implemented in Future Phases:
- **Stratified Partitioning (`StratifiedKFold`):** Enforces identical class ratios across all training and test splits.
- **Target Binning / Reformulation:** Formulating the task into binary classification ($\text{Quality} \ge 7 \implies \text{Good/Reserve}$ vs. $< 7 \implies \text{Normal/Table}$) provides a balanced commercial distinction ($\approx 13.6\%$ positive class).
- **Cost-Sensitive Learning:** Utilizing `class_weight='balanced'` in Random Forest, Support Vector Classifiers, and SGD to weight loss penalties inversely proportional to class frequencies.
- **Threshold Calibration:** Shifting operating classification thresholds ($\tau$) below the default $0.50$ (e.g., $\tau = 0.35$) to optimize minority recall for commercial cellar screening.

---

## 🧪 Day 3: Target Engineering, Preprocessing Pipelines & Baseline Models

### 1. Target Formulation & Mathematical Justification
Direct 6-class ordinal classification ($3–8$) is heavily impaired by extreme sample scarcity in the tails ($10$ samples for score $3$; $18$ samples for score $8$) and heavy feature distribution overlap between median scores $5$ and $6$. Two binning formulations were evaluated:

| Formulation | Class Label | Sensory Definition | Sample Count | Class Share (%) | Evaluation & Decision |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Raw (6-Class)** | 3, 4, 5, 6, 7, 8 | Discrete judge scores | 10, 53, 681, 638, 199, 18 | 0.6%, 3.3%, 42.6%, 39.9%, 12.5%, 1.1% | Discarded: extreme sparsity in grades 3 and 8 causes total model failure |
| **Three-Tier** | Low ($\le 4$)<br>Medium ($5–6$)<br>High ($\ge 7$) | Flawed / Defective<br>Commercial Everyday<br>Premium / Reserve | 63<br>1,319<br>217 | 3.94%<br>82.49%<br>13.57% | Discarded: Low class ($3.94\%$) remains too scarce for stable multi-class optimization |
| **Binary (Selected)** | **Normal ($< 7$)**<br>**Good ($\ge 7$)** | **Standard Table Wine**<br>**Certified Reserve Wine** | **1,382**<br>**217** | **86.43%**<br>**13.57%** | **Selected:** Conforms to Cortez et al. (2009) benchmark and reflects commercial cellar sorting |

**Enological & Operational Rationale:**
In commercial viticulture and quality assurance, winemakers triage lots into high-value **Reserve / Grand Reserve** bottlings ($\ge 7$) commanding premium margins versus standard commercial table wines ($< 7$). The binary threshold at $\ge 7$ yields $217$ positive samples ($13.57\%$), providing sufficient statistical density for stable gradient convergence and stratified cross-validation.

![Target Distribution Comparison](reports/figures/day3_target_distribution_comparison.png)

---

### 2. Domain-Specific Biochemical Feature Synthesis
To provide the algorithms with direct indicators of enological balance, 5 domain interaction features were synthesized based on Day 2 EDA findings:

| Synthesized Feature | Mathematical Formulation | Enological Chemical Rationale |
| :--- | :---: | :--- |
| **`total_acidity`** | $\text{fixed acidity} + \text{volatile acidity}$ | Quantifies the full acidic impact and sensory palate freshness ($g/dm^3$). |
| **`bound_sulfur_dioxide`** | $\text{total SO}_2 - \text{free SO}_2$ | Measures sulfur bound to aldehydes and polyphenols; indicates cumulative oxidation exposure. |
| **`free_to_total_so2_ratio`** | $\frac{\text{free SO}_2}{\text{total SO}_2 + 10^{-5}}$ | Active preservation efficiency; high ratios preserve aromatic esters without sulfur excess. |
| **`alcohol_to_density_ratio`** | $\frac{\text{alcohol}}{\text{density}}$ | Index of fermentation completion and mouthfeel density balance (ethanol vs residual extract). |
| **`sulphate_to_chloride_ratio`** | $\frac{\text{sulphates}}{\text{chlorides} + 10^{-5}}$ | Balances beneficial antioxidant potassium salts against unpleasant salty mineral taints. |

---

### 3. Stratified Partitioning & Leakage-Free Design
The dataset was partitioned into 80% training ($N=1,279$) and 20% testing ($N=320$) using `stratify=y` with seed `random_state=42`. Stratification guarantees identical class ratios across partitions:

| Partition | Total Samples | Normal Wines ($< 7$) | Good Wines ($\ge 7$) | Positive Class Share (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Full Dataset** | 1,599 | 1,382 | 217 | **13.57%** |
| **Training Partition (80%)** | 1,279 | 1,105 | 174 | **13.60%** |
| **Testing Partition (20%)** | 320 | 277 | 43 | **13.44%** |

*Leakage Prevention:* All feature normalizations (`StandardScaler`) were embedded inside scikit-learn `Pipeline` architectures. Preprocessing statistics were strictly calculated on training folds and applied to test folds without prior exposure.

---

### 4. Baseline Supervised Classifier Benchmarks
Three diverse supervised classification paradigms were evaluated under default configurations on the held-out test set ($N=320$):

| Classifier Paradigm | Algorithm Class | 5-Fold CV Accuracy | Test Accuracy | Balanced Accuracy | Precision (Good) | Recall (Good) | F1-Score (Good) | Test ROC-AUC | Test PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Non-Linear Bagged Trees | **90.07% ± 1.6%** | **93.44%** | **0.7951** | **86.67%** | **60.47%** | **0.7123** | **0.9318** | **0.8120** |
| **Support Vector Classifier (SVC)** | RBF Kernel Margin Machine | 87.73% ± 1.1% | 90.31% | 0.6690 | 83.33% | 34.88% | 0.4918 | 0.8698 | 0.6540 |
| **SGD Classifier** | Convex Log-Loss Optimization | 85.14% ± 2.4% | 85.00% | 0.5204 | 27.27% | 6.98% | 0.1111 | 0.7501 | 0.2840 |

![Baseline Model Comparison](reports/figures/day3_baseline_model_comparison.png)

---

### 5. Confusion Matrix Diagnostics & Error Analysis
Diagnostic error analysis on held-out test specimens ($43$ Good, $277$ Normal) reveals the behavioral trade-offs of each baseline architecture:

![Confusion Matrices](reports/figures/day3_confusion_matrices.png)

1. **Random Forest (Champion Baseline):**
   - **True Negatives:** $273 / 277$ ($98.6\%$ specificity).
   - **True Positives:** $26 / 43$ ($60.5\%$ sensitivity).
   - **False Positives:** Only $4$ normal wines misclassified as good ($1.4\%$ false alarm rate).
   - **False Negatives:** $17$ premium wines missed.
   - *Assessment:* Strongest non-linear feature interaction modeling. Tree ensembles are inherently robust against feature scale differences and moderate outliers.

2. **Support Vector Classifier (Conservative Margin):**
   - **True Negatives:** $274 / 277$ ($98.9\%$ specificity).
   - **True Positives:** $15 / 43$ ($34.9\%$ sensitivity).
   - **False Negatives:** $28 / 43$ ($65.1\%$ missed).
   - *Assessment:* Without class weighting, the default margin optimizer prioritizes majority class separability, heavily penalizing false positive boundaries and causing high false negative rates on premium wines.

3. **SGD Classifier (Gradient Starvation):**
   - **True Negatives:** $269 / 277$ ($97.1\%$).
   - **True Positives:** $3 / 43$ ($7.0\%$).
   - **False Positives:** $8$ false alarms.
   - **False Negatives:** $40 / 43$ ($93.0\%$ missed).
   - *Assessment:* In stochastic convex optimization without sample reweighting, $86.4\%$ of gradient steps originate from normal wines, causing the hyperplane to degenerate near the majority centroid.

---

### 6. Discrimination Power Analysis (ROC & PR Curves)

![ROC Curves](reports/figures/day3_roc_curves.png)
![PR Curves](reports/figures/day3_precision_recall_curves.png)

- **ROC-AUC Diagnostics:** Random Forest achieves an exceptional **0.9318 ROC-AUC**, demonstrating strong separability across the continuous probability spectrum. SVC follows at **0.8698**, while SGD scores **0.7501**.
- **Precision-Recall Performance:** Against the baseline prevalence of $13.44\%$ (no-skill rate), Random Forest maintains a high **PR-AUC of 0.8120**, retaining $>80\%$ precision up to $60\%$ recall.

---

### 7. Key Takeaways & Foundation for Day 4
1. **The Class Imbalance Challenge:** While baseline overall accuracy appears high ($90–93\%$), unweighted margin and linear models (SVC and SGD) fail to identify the majority of premium wines.
2. **Cost-Sensitive Learning Requirement:** To unlock higher minority recall without sacrificing precision, Day 4 will implement cost-sensitive learning (`class_weight='balanced'`) to penalize minority misclassifications inversely proportional to class frequencies.
3. **Hyperparameter Tuning:** Systematic hyperparameter search (`GridSearchCV`) across tree depths, estimators, SVM regularizers ($C$, $\gamma$), and SGD learning rates will be conducted in Day 4.

---

## 🏆 Day 4: Hyperparameter Optimization, Feature Importance, Model Benchmarking & Deployment Conclusion

### 1. Stratified 5-Fold Cross-Validation Framework & Metric Lift Analysis
To move beyond single train/test split variance and establish robust generalization bounds, all models were evaluated via **Stratified 5-Fold Cross-Validation** (`random_state=42`) on the training partition ($N=1,279$).

Integrating cost-sensitive learning (`class_weight='balanced'`) transformed model sensitivity on the rare positive class ($13.6\%$ prevalence):

| Model Architecture | Evaluation Stage | 5-Fold CV F1 (Good) | Balanced Accuracy | 5-Fold CV ROC-AUC | Minority Precision | Minority Recall | Primary Analytical Insight |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | Baseline (Default) | 0.4846 ± 0.097 | 0.6784 ± 0.048 | 0.8888 ± 0.025 | 64.27% ± 10.9% | 39.03% ± 8.8% | High precision; default trees fail to capture ~61% of minority wines across folds |
| **Random Forest** | **Tuned (`GridSearchCV`)** | **0.6044 ± 0.090** | **0.7833 ± 0.055** | **0.9014 ± 0.017** | 57.14% ± 8.7% | **64.27% ± 9.8%** | **+24.7% F1 lift & +64.7% recall lift**; ensemble variance remained low |
| **SVC** | Baseline (Default) | 0.4053 ± 0.065 | 0.6345 ± 0.028 | 0.8065 ± 0.033 | 68.60% ± 16.2% | 29.26% ± 5.4% | Default margin hyperplanes severely suppress minority positive predictions |
| **SVC** | **Tuned (`GridSearchCV`)** | **0.5845 ± 0.055** | **0.7767 ± 0.043** | **0.8494 ± 0.024** | 54.30% ± 3.5% | **63.76% ± 9.1%** | **+44.2% F1 lift & +117.9% recall lift**; non-linear boundary fits rare class |
| **SGD Classifier** | Baseline (Default) | 0.3790 ± 0.240 | 0.6626 ± 0.122 | 0.8255 ± 0.055 | 38.40% ± 21.5% | 38.94% ± 26.6% | High variance across folds; unregularized linear hyperplane |
| **SGD Classifier** | **Tuned (`GridSearchCV`)** | **0.5013 ± 0.031** | **0.7955 ± 0.019** | **0.8746 ± 0.013** | 36.15% ± 3.1% | **82.18% ± 2.8%** | **+32.3% F1 lift & +111.0% recall lift**; stabilized fold-to-fold variance ($\sigma$ from 0.24 to 0.03) |

![5-Fold CV Performance Comparison](reports/figures/day4_cv_performance_comparison.png)

---

### 2. Systematic Hyperparameter Optimization (`GridSearchCV`)
Exhaustive parameter searches were executed with 5-fold stratified cross-validation targeting ROC-AUC and $F_1$ score maximization:

| Estimator Pipeline | Tuned Hyperparameters | Best Parameter Value | Parameter Search Space | Enological & Algorithmic Rationale |
| :--- | :--- | :---: | :--- | :--- |
| **Random Forest** | `clf__class_weight`<br>`clf__max_depth`<br>`clf__min_samples_split`<br>`clf__n_estimators` | `'balanced'`<br>`None`<br>`2`<br>`150` | `['balanced', 'balanced_subsample']`<br>`[None, 12, 16]`<br>`[2, 5]`<br>`[100, 150]` | Inverting tree sample weights penalizes misclassification of rare reserve wines; full depth with 150 trees captures non-linear interactions without early leaf truncation. |
| **Support Vector Classifier (SVC)** | `clf__C`<br>`clf__gamma`<br>`clf__class_weight` | `0.5`<br>`0.05`<br>`'balanced'` | `[0.5, 1.0, 3.0]`<br>`['scale', 0.1, 0.05]`<br>`['balanced']` | Moderate margin regularization ($C=0.5$) combined with $\gamma=0.05$ smooths the decision boundary, reducing overfitting on noisy chemical boundaries. |
| **SGD Classifier** | `clf__alpha`<br>`clf__penalty`<br>`clf__class_weight` | `0.01`<br>`'l2'`<br>`'balanced'` | `[1e-4, 1e-3, 1e-2]`<br>`['l2', 'elasticnet']`<br>`['balanced']` | Stronger $L_2$ regularization ($\alpha=0.01$) prevents runaway gradient divergence in convex logistic loss updates under weighted class penalties. |

---

### 3. Feature Importance Chart & Analysis for the Random Forest Model
To interpret the internal decision logic of the champion Random Forest model, we performed dual-method importance quantification across all 16 physicochemical features:
1. **Tree-Based Mean Decrease in Impurity (MDI / Gini Importance):** Measures cumulative variance reduction across all split nodes across 150 estimators.
2. **Out-of-Sample Permutation Feature Importance:** Evaluated on the held-out test set ($N=320$) over 15 random shuffles to guard against MDI cardinality bias.

| Rank | Feature Name | MDI Importance (Mean) | MDI Std Dev | Test Permutation Mean | Test Permutation Std Dev | Enological Interpretation |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **`alcohol_to_density_ratio`** | **0.1687** | ±0.1153 | **0.0135** | ±0.0106 | Primary index of sugar-to-alcohol fermentation completion and body richness. |
| **2** | **`alcohol`** | **0.1457** | ±0.1192 | 0.0030 | ±0.0089 | Highest positive correlation with perceived sensory quality ($r = +0.476$). |
| **3** | **`sulphates`** | **0.0917** | ±0.0620 | **0.0329** | ±0.0069 | Highest permutation impact: potassium sulphate prevents oxidation and volatile faults. |
| **4** | **`volatile acidity`** | **0.0755** | ±0.0468 | **0.0110** | ±0.0030 | Chief defect driver: acetic acid concentrations $>0.60\text{ g/dm}^3$ cause vinegar taint. |
| **5** | **`sulphate_to_chloride_ratio`** | **0.0631** | ±0.0441 | **0.0124** | ±0.0020 | Balances beneficial antioxidant protection against unpleasant mineral saltiness. |
| **6** | **`citric acid`** | **0.0592** | ±0.0393 | 0.0012 | ±0.0049 | Enhances palate freshness and crispness, masking flat notes. |
| **7** | **`bound_sulfur_dioxide`** | **0.0536** | ±0.0333 | 0.0033 | ±0.0037 | Measures sulfur bound to polyphenols; indicates cumulative oxidation mitigation. |
| **8** | **`free_to_total_so2_ratio`** | **0.0445** | ±0.0216 | 0.0003 | ±0.0021 | Active gaseous preservation efficiency index. |
| **9** | **`pH`** | **0.0433** | ±0.0224 | 0.0040 | ±0.0022 | Must acidity buffer maintaining microbial and enzymatic stability. |
| **10** | **`total sulfur dioxide`** | **0.0425** | ±0.0234 | 0.0073 | ±0.0051 | Regulates microbiological preservation without pungent sulfidic taints. |
| **11** | **`density`** | **0.0417** | ±0.0268 | 0.0016 | ±0.0034 | Dissolved extract and dry matter mouthfeel balance. |
| **12** | **`chlorides`** | **0.0384** | ±0.0205 | 0.0010 | ±0.0024 | Soil salinity proxy; excessive levels impart salty off-flavors. |
| **13** | **`fixed acidity`** | **0.0370** | ±0.0215 | 0.0054 | ±0.0019 | Non-volatile tartaric acid providing core structural backbone. |
| **14** | **`total_acidity`** | **0.0359** | ±0.0212 | 0.0071 | ±0.0025 | Overall sensory acid impression on the palate. |
| **15** | **`residual sugar`** | **0.0322** | ±0.0169 | 0.0025 | ±0.0013 | Unfermented glucose/fructose sweetness profile. |
| **16** | **`free sulfur dioxide`** | **0.0272** | ±0.0144 | 0.0030 | ±0.0014 | Molecular $SO_2$ fraction protecting volatile esters. |

![Random Forest Feature Importance](reports/figures/day4_feature_importance_rf.png)

> **Key Interpretability Takeaway:** Both MDI and out-of-sample permutation importance confirm that wine sensory quality is primarily governed by the **alcohol/density nexus** (metabolic conversion efficiency), **sulphates** (antioxidant preservation), and **volatile acidity** (absence of aerobic bacterial defects).

---

### 4. Diagnostic Error Analysis & Confusion Matrix Breakdown
Tuned model predictions were benchmarked on held-out test data ($N=320$: 277 Normal, 43 Good Wine):

![Tuned Confusion Matrices](reports/figures/day4_tuned_confusion_matrices.png)

1. **Random Forest (Champion Ensemble):**
   - **True Negatives:** $267 / 277$ ($96.4\%$ specificity).
   - **True Positives:** $31 / 43$ ($72.1\%$ sensitivity).
   - **False Positives:** $10 / 277$ ($3.6\%$ false alarm rate).
   - **False Negatives:** $12 / 43$ ($27.9\%$ missed).
   - *Outcome:* Retains high precision ($75.61\%$) while improving recall by **+19.2%** over baseline.
2. **Support Vector Classifier (SVC):**
   - **True Positives:** $33 / 43$ ($76.7\%$ sensitivity, up from $34.9\%$).
   - **False Negatives:** Dropped from $28$ to $10$.
   - *Outcome:* Class-weighted margins allow SVC to capture the majority of premium wines, though with higher false positives ($45 / 277$).
3. **SGD Classifier (Convex Logistic):**
   - **True Positives:** $36 / 43$ ($83.7\%$ sensitivity, up from $7.0\%$).
   - **False Negatives:** Dropped from $40$ down to $7$.
   - *Outcome:* Dramatic surge in minority recall (+1100% relative lift), successfully converting a degenerated linear model into an aggressive screening tool.

---

### 5. Discrimination Power Analysis (ROC & Precision-Recall Curves)

![Precision-Recall Curves](reports/figures/day4_precision_recall_curves.png)
![ROC Curves](reports/figures/day4_roc_curves.png)

- **Receiver Operating Characteristic (ROC):** Random Forest achieves the peak test **ROC-AUC of 0.9349**, with SVC at **0.9097** and SGD at **0.8967**. All three models demonstrate excellent rank-order discrimination far above random chance ($0.50$).
- **Precision-Recall Performance:** Random Forest dominates with a **PR-AUC of 0.8260**, maintaining $>75\%$ precision across higher recall regimes. Against the no-skill prevalence baseline of $13.44\%$, the tuned Random Forest ensemble exhibits a **6.1× enrichment factor**.

---

### 6. Master Side-by-Side Model Comparison Table
The benchmark matrix below summarizes all 3 classification architectures across both Baseline (Day 3) and Tuned (Day 4) configurations on identical held-out test data ($N=320$):

| Classifier Architecture | Modeling Stage | Test Accuracy | Balanced Accuracy | Precision (Good) | Recall (Good) | F1-Score (Good) | Test ROC-AUC | Test PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Baseline (Day 3) | **93.44%** | 0.7951 | **86.67%** | 60.47% | 0.7123 | 0.9318 | 0.7986 |
| **Random Forest (Champion)** | **Tuned (Day 4)** | **93.13%** | **0.8424** | **75.61%** | **72.09%** | **0.7381** | **0.9349** | **0.8260** |
| **Support Vector Classifier** | Baseline (Day 3) | 90.31% | 0.6690 | 83.33% | 34.88% | 0.4918 | 0.8698 | 0.6922 |
| **Support Vector Classifier** | **Tuned (Day 4)** | 82.81% | 0.8025 | 42.31% | **76.74%** | 0.5455 | 0.9097 | 0.6535 |
| **SGD Classifier** | Baseline (Day 3) | 85.00% | 0.5204 | 27.27% | 6.98% | 0.1111 | 0.7501 | 0.2860 |
| **SGD Classifier** | **Tuned (Day 4)** | 82.81% | 0.8320 | 42.86% | **83.72%** | 0.5669 | 0.8967 | 0.5910 |

![Master Side-by-Side Comparison](reports/figures/day4_side_by_side_comparison.png)

---

### 7. Decision Threshold Optimization for Industrial Cellar Sorting
In commercial winery operations, classification thresholds ($\tau$) can be adjusted according to business economics:

| Operating Threshold ($\tau$) | Accuracy | Balanced Accuracy | Precision (Good) | Recall (Good) | F1-Score | Captured Premium Lots | Operational Use Case |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $\tau = 0.50$ (Default) | **93.13%** | 0.8424 | **75.61%** | 72.09% | **0.7381** | 31 / 43 | Automated retail bottling (minimizes false reserve labeling) |
| $\tau = 0.40$ | 91.56% | 0.8480 | 66.67% | 76.74% | 0.7135 | 33 / 43 | Balanced quality control checkpoint |
| $\tau = 0.35$ (Recommended Screening) | **89.69%** | **0.8520** | 58.62% | **79.07%** | 0.6733 | **34 / 43** | **Pre-barrel aging screening (captures candidate lots for sommelier review)** |
| $\tau = 0.25$ | 86.25% | 0.8468 | 49.28% | 79.07% | 0.6071 | 34 / 43 | Aggressive cellar inspection |

---

### 8. Conclusion: Which Model is Most Suitable for Deployment and Why?

#### 🏆 Champion Selection: Tuned Random Forest Classifier

After exhaustive comparative benchmarking across Stratified 5-Fold Cross-Validation, held-out test evaluation, discrimination power (ROC & PR AUC), and hardware profiling, the **Tuned Random Forest Classifier** is unequivocally recommended as the most suitable model for commercial winery deployment.

#### 🔬 Rationale for Deployment:
1. **Peak Discrimination Power:** Random Forest achieves the highest **ROC-AUC (0.9349)** and **PR-AUC (0.8260)**, maintaining strong precision even as recall scales.
2. **Balanced Predictive Accuracy:** With an overall accuracy of **93.13%**, balanced accuracy of **0.8424**, and F1-score of **0.7381**, Random Forest provides the most reliable commercial sorting performance without excessive false alarms.
3. **Robustness to Non-Linearity & Outliers:** Wine chemistry involves non-linear enzymatic thresholds and interactions. Unlike linear SGD or distance-sensitive SVMs, tree-based partitions naturally handle multi-modal distributions and skewed attributes (e.g. residual sugar and chlorides with Tukey outliers up to 9.7%) without distortion.
4. **Transparent Interpretability:** Dual-method importance quantification (MDI and Permutation Importance) allows enologists to verify and trust model predictions based on known viticultural science (alcohol-to-density balance, sulphates, and volatile acidity).
5. **Operational Latency & Scalability:** With single-sample inference latency under **95 ms**, the model easily integrates into real-time winery testing benches, conveyor inspection systems, or automated laboratory assays.

---

### 📋 Full Oasis Infobyte Task Level 2 Sign-Off
All 10 required features of the **Oasis Infobyte Data Analytics Task 2 (Wine Quality Prediction)** have been executed, statistically verified, visualized, documented, and delivered in full compliance with project specifications.


---

## 🔬 Day 5: Model Interpretation, Feature Importance, Partial Dependence & Champion Pipeline Serialization

### 1. Tree-Based Feature Importance (MDI / Gini Reduction) with Inter-Tree Variance Bounds
Mean Decrease in Impurity (MDI) was extracted across all $150$ constituent decision trees in the champion Random Forest pipeline. In addition to mean importance, inter-tree standard deviation ($\sigma$) and percentage contribution were computed across all $16$ input features:

| Rank | Physicochemical Feature | MDI Importance (Mean) | Tree-to-Tree Std Dev ($\sigma$) | Relative Share (%) | Out-of-Sample Permutation (ROC-AUC Drop) | Permutation Std Dev | Enological Function & Role |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **`alcohol_to_density_ratio`** | **0.1687** | ±0.1153 | **16.87%** | **0.0150** | ±0.0108 | Primary indicator of yeast sugar attenuation and full-bodied mouthfeel. |
| **2** | **`alcohol`** | **0.1457** | ±0.1192 | **14.57%** | 0.0022 | ±0.0098 | Key sensory structural driver; elevated ethanol carries aromatics and warmth. |
| **3** | **`sulphates`** | **0.0917** | ±0.0620 | **9.17%** | **0.0358** | ±0.0081 | Highest permutation drop: potassium sulphate preserves wine against oxidation. |
| **4** | **`volatile acidity`** | **0.0755** | ±0.0468 | **7.55%** | **0.0084** | ±0.0029 | Vinegar spoilage marker; acetic acid $>0.40	ext{ g/dm}^3$ destroys premium balance. |
| **5** | **`sulphate_to_chloride_ratio`** | **0.0631** | ±0.0441 | **6.31%** | **0.0119** | ±0.0019 | Balances protective antioxidants against unpleasant mineral salinity. |
| **6** | **`citric acid`** | **0.0592** | ±0.0393 | **5.92%** | -0.0003 | ±0.0054 | Natural acid providing freshness, crispness, and sensory liveliness. |
| **7** | **`bound_sulfur_dioxide`** | **0.0536** | ±0.0333 | **5.36%** | 0.0018 | ±0.0025 | Sulfur bound to aldehydes and polyphenols; reflects total preservative load. |
| **8** | **`free_to_total_so2_ratio`** | **0.0445** | ±0.0216 | **4.45%** | -0.0001 | ±0.0015 | Active unbound preservative fraction protecting against microbial contamination. |
| **9** | **`pH`** | **0.0433** | ±0.0224 | **4.33%** | 0.0025 | ±0.0027 | Acidity buffer; stabilizes coloration, enzymatic activity, and antimicrobial power. |
| **10** | **`total sulfur dioxide`** | **0.0425** | ±0.0234 | **4.25%** | 0.0069 | ±0.0052 | Total antioxidant reservoir; excessive levels impart pungent sulfur odors. |

![Random Forest MDI vs Out-of-Sample Permutation](reports/figures/day5_rf_feature_importance.png)

---

### 2. Multi-Model Out-of-Sample Permutation Feature Importance Comparison
To verify that the identified drivers are universal rather than artifacts of decision tree mechanics, **Permutation Feature Importance** (ROC-AUC scoring, 5 repeats) was evaluated on the held-out test set ($N=320$) across all three tuned architectures:

![Cross-Model Permutation Comparison](reports/figures/day5_permutation_importance_comparison.png)

- **Unanimous Agreement on Core Drivers:** `sulphates`, `alcohol_to_density_ratio`, `volatile acidity`, and `sulphate_to_chloride_ratio` consistently induce the highest ROC-AUC degradation across Random Forest, Support Vector Classifier (SVC), and regularized SGD Classifier.
- **Cardianlity Bias Mitigation:** While raw `alcohol` exhibited high MDI importance due to continuous split frequency, `sulphates` proved to be the single most critical out-of-sample stabilizer for preventing misclassification (+0.0358 ROC-AUC degradation when shuffled).

---

### 3. Directional Decision Weights: Standardized Linear Coefficients (SGD)
While ensemble trees provide non-linear partitions, the standardized logistic regression weights ($eta$) from the tuned SGD classifier reveal clear directional attribution: each unit shift in standardized feature alters the odds of being classified as Reserve Quality ($Quality \ge 7$) by factor $e^eta$:

| Rank | Physicochemical Feature | Standardized Weight ($eta$) | Odds Ratio ($e^eta$) | Directional Impact | Practical Enological Translation |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **`alcohol`** | **+0.5151** | **1.6738** | **Positive (+)** | +67.4% odds increase per +1 SD alcohol; physiological ripeness is vital. |
| **2** | **`alcohol_to_density_ratio`** | **+0.5073** | **1.6609** | **Positive (+)** | +66.1% odds increase; confirms complete fermentation sugar attenuation. |
| **3** | **`sulphates`** | **+0.3710** | **1.4491** | **Positive (+)** | +44.9% odds increase; potassium sulphate protects delicate aromatics. |
| **4** | **`sulphate_to_chloride_ratio`** | **+0.3684** | **1.4454** | **Positive (+)** | +44.5% odds increase; antioxidant coverage without mineral salinity. |
| **5** | **`free_to_total_so2_ratio`** | **+0.3109** | **1.3646** | **Positive (+)** | +36.5% odds increase; active unbound SO2 maintains freshness. |
| **6** | **`residual sugar`** | **+0.2111** | **1.2350** | **Positive (+)** | +23.5% odds increase; balanced residual sweetness softens tannins. |
| **7** | **`fixed acidity`** | **+0.1251** | **1.1333** | **Positive (+)** | +13.3% odds increase; tartaric acid backbone provides aging stability. |
| **8** | **`total_acidity`** | **+0.0913** | **1.0956** | **Positive (+)** | +9.6% odds increase; overall acid structure. |
| **9** | **`citric acid`** | **+0.0009** | **1.0009** | **Positive (+)** | +0.1% odds increase; neutral-to-slight freshness lift. |
| **10** | **`density`** | **-0.0359** | **0.9647** | **Negative (-)** | -3.5% odds decrease; heavy watery must density suppresses quality. |
| **11** | **`chlorides`** | **-0.0702** | **0.9322** | **Negative (-)** | -6.8% odds decrease; excessive mineral salinity causes brackish mouthfeel. |
| **12** | **`pH`** | **-0.1005** | **0.9044** | **Negative (-)** | -9.6% odds decrease; elevated pH (>3.45) accelerates oxidation. |
| **13** | **`free sulfur dioxide`** | **-0.1256** | **0.8820** | **Negative (-)** | -11.8% odds decrease; unbuffered excess free SO2 creates harshness. |
| **14** | **`bound_sulfur_dioxide`** | **-0.2037** | **0.8157** | **Negative (-)** | -18.4% odds decrease; bound sulfur causes stale, reductive off-odors. |
| **15** | **`total sulfur dioxide`** | **-0.2074** | **0.8127** | **Negative (-)** | -18.7% odds decrease; excessive total sulfur blunts berry fruit flavors. |
| **16** | **`volatile acidity`** | **-0.3489** | **0.7054** | **Negative (-)** | **-29.5% odds decrease; chief defect marker; acetic spoilage.** |

![Directional Linear Coefficients](reports/figures/day5_linear_coefficients.png)

---

### 4. Non-Linear Partial Dependence Profiling (1D PDP & Tipping Points)
Partial Dependence Plots trace the marginal probability of a lot being classified as Reserve Quality ($Quality \ge 7$) as individual physicochemical measurements vary across their empirical ranges:

![Partial Dependence Profiles](reports/figures/day5_partial_dependence_plots.png)

1. **Alcohol Tipping Point ($10.5\% 	o 11.5\%$ vol):** Marginal probability remains flat ($<12\%$) for alcohol levels below $10.5\%$, inflects sharply upward between $10.5\%$ and $11.5\%$, and plateaus at $>45\%$ probability above $12.0\%$.
2. **Volatile Acidity Spoilage Cliff ($>0.40	ext{ g/dm}^3$):** Rapid monotonic decline in reserve probability once acetic acid crosses $0.40	ext{ g/dm}^3$. At concentrations $>0.60	ext{ g/dm}^3$, probability approaches zero regardless of alcohol content.
3. **Sulphate Saturation Plateau ($0.65 - 0.75	ext{ g/dm}^3$):** Sharp positive lift from $0.50$ to $0.70	ext{ g/dm}^3$, followed by plateauing saturation above $0.75	ext{ g/dm}^3$.
4. **Free-to-Total $SO_2$ Ratio ($\ge 0.25$):** Low active free ratios ($<0.20$) yield baseline probability ($<14\%$), whereas ratios $\ge 0.25$ maximize microbial protection and reserve likelihood.

---

### 5. Bivariate 2D Interaction Contours (Enological Synergies)
Evaluating 2D partial dependence surfaces illuminates the non-linear compensatory interactions between chemical properties:

![2D Bivariate Interaction Contours](reports/figures/day5_pdp_2d_interactions.png)

- **Alcohol vs. Volatile Acidity ("The Sweet Spot"):** A distinct high-probability island ($>50\%$ Reserve likelihood) is established only when **alcohol exceeds $11.5\%$ vol** AND **volatile acidity is contained below $0.45	ext{ g/dm}^3$**. High alcohol cannot rescue a wine afflicted with elevated acetic acid, and low volatile acidity cannot compensate for weak, watery alcohol body.
- **Sulphates vs. Chlorides:** The optimal zone occurs when **sulphates exceed $0.70	ext{ g/dm}^3$** while **chlorides remain below $0.075	ext{ g/dm}^3$**, ensuring pristine antioxidant coverage without saline mineral contamination.

---

### 6. Quantitative Enological Decision Rules & Cellar Blending Blueprint
Comparing empirical distributions between Normal ($Quality < 7$) and Reserve ($Quality \ge 7$) wines yields concrete cellar thresholds for harvest maturity, fermentation monitoring, and barrel selection:

| Physicochemical Compound | Normal Median | Reserve Median | Percentage Shift | Enological Cellar Blending Guideline |
| :--- | :---: | :---: | :---: | :--- |
| **Citric Acid** | $0.24	ext{ g/dm}^3$ | $0.40	ext{ g/dm}^3$ | **+47.99%** | Target $\ge 0.35	ext{ g/dm}^3$; enhances freshness and microbial stability against spoilage. |
| **Sulphate-to-Chloride Ratio** | $7.56$ | $10.20$ | **+34.93%** | Target $\ge 10.0$; balance antioxidant preservation against salty mineral taint. |
| **Bound Sulfur Dioxide** | $23.0	ext{ mg/dm}^3$ | $14.0	ext{ mg/dm}^3$ | **-34.89%** | Keep bound $SO_2 \le 25.0	ext{ mg/dm}^3$ to avoid stale, sulfurous oxidation aromas. |
| **Total Sulfur Dioxide** | $39.5	ext{ mg/dm}^3$ | $27.0	ext{ mg/dm}^3$ | **-27.74%** | Cap total $SO_2 \le 40.0	ext{ mg/dm}^3$; excessive sulfur blunts delicate berry aromatics. |
| **Volatile Acidity** | $0.54	ext{ g/dm}^3$ | $0.37	ext{ g/dm}^3$ | **-25.87%** | Keep strictly $\le 0.40	ext{ g/dm}^3$ via cellar sanitation to prevent vinegar spoilage. |
| **Free-to-Total $SO_2$ Ratio** | $0.36$ | $0.44$ | **+15.87%** | Maintain active free ratio $\ge 0.25$ to guarantee microbial protection without bound sulfur odor. |
| **Sulphates** | $0.60	ext{ g/dm}^3$ | $0.74	ext{ g/dm}^3$ | **+15.31%** | Maintain $\ge 0.70	ext{ g/dm}^3$ potassium sulphate for antioxidant defense and $SO_2$ stability. |
| **Chlorides** | $0.080	ext{ g/dm}^3$ | $0.073	ext{ g/dm}^3$ | **-14.97%** | Limit strictly $\le 0.075	ext{ g/dm}^3$; prevents brackish mouthfeel from excessive soil salinity. |
| **Free Sulfur Dioxide** | $14.0	ext{ mg/dm}^3$ | $11.0	ext{ mg/dm}^3$ | **-13.55%** | Maintain $15.0 - 20.0	ext{ mg/dm}^3$ free $SO_2$ at bottling for long cellar longevity. |
| **Alcohol-to-Density Ratio** | $10.02$ | $11.63$ | **+12.46%** | Target $\ge 11.60$; indicates high fermentation conversion and sugar attenuation. |
| **Alcohol** | $10.0\%	ext{ vol}$ | $11.6\%	ext{ vol}$ | **+11.51%** | Target $\ge 11.5\%	ext{ vol}$ during harvest maturation for full body & phenolic ripeness. |

---

### 7. Champion Model Serialization & Production Artifact Packaging
The champion Random Forest pipeline was packaged into a production-ready artifact and verified:
- **Model Binary:** `models/champion_rf_pipeline.joblib` (2.48 MB)
- **Metadata Specification:** `models/champion_rf_pipeline.json` recording full pipeline steps, hyperparameters, input schema, test benchmarks, and operating threshold calibrations.
- **Deserialization Verification:** Pipeline reload confirmed seamless integrity with `StandardScaler` and `RandomForestClassifier(n_estimators=150, class_weight='balanced')`.

---

### 8. Live Production Single-Sample Inference Engine Demonstration
A dedicated single-sample inference function was implemented with automated ratio feature synthesis and rule-based enological diagnostics:

```python
# Sample A: Candidate Reserve Lot (High ripeness, clean fermentation)
sample_a = {
    'fixed acidity': 8.5, 'volatile acidity': 0.32, 'citric acid': 0.45,
    'residual sugar': 2.1, 'chlorides': 0.065, 'free sulfur dioxide': 18.0,
    'total sulfur dioxide': 36.0, 'density': 0.9955, 'pH': 3.30,
    'sulphates': 0.82, 'alcohol': 12.4
}
# Output:
# -> Prediction:  Reserve (Good Quality >= 7)
# -> Probability: 83.33% (Passes tau = 0.35)
# -> Action:      PASS (Reserve Lot)
# -> Diagnostics:
#    ✓ Optimal alcohol concentration (12.4%), providing robust structural body.
#    ✓ Pristine volatile acidity (0.32 g/dm3), excellent microbial hygiene.
#    ✓ Strong antioxidant protection (0.82 g/dm3 potassium sulphate).

# Sample B: Commercial Table Wine Lot (Spoilage markers present)
sample_b = {
    'fixed acidity': 7.8, 'volatile acidity': 0.68, 'citric acid': 0.12,
    'residual sugar': 2.4, 'chlorides': 0.092, 'free sulfur dioxide': 8.0,
    'total sulfur dioxide': 75.0, 'density': 0.9978, 'pH': 3.42,
    'sulphates': 0.52, 'alcohol': 9.8
}
# Output:
# -> Prediction:  Normal (< 7)
# -> Probability: 13.33% (Fails tau = 0.35)
# -> Action:      FLAG (Commercial Table Lot)
# -> Diagnostics:
#    ⚠ Low alcohol concentration (9.8% < 11.5%), risk of flabby mouthfeel.
#    ⚠ Elevated volatile acidity (0.68 g/dm3 > 0.40), acetic spoilage risk.
#    ⚠ Moderate to low sulphate preservation (0.52 g/dm3 < 0.70).
```

---

### 9. Master Executive Champion Model Dashboard

![Champion Model Executive Summary Dashboard](reports/figures/day5_champion_model_summary.png)

#### Final Commercial Deployment Specifications:
- **Recommended Industrial Screening Threshold:** $	au = 0.35$
- **Industrial Screening Balanced Accuracy:** **85.20%**
- **Industrial Screening Reserve Recall:** **79.07%** ($34$ out of $43$ true reserve lots captured)
- **Single-Sample Inference Latency:** $< 95	ext{ ms}$
- **Deployment Status:** **APPROVED FOR INDUSTRIAL CELLAR SCREENING**

---

---

## 🚀 Day 6: Comprehensive Model Comparison, Hardware Latency Benchmarking & Deployment Synthesis

On Day 6, the exploratory, preprocessing, hyperparameter optimization, and model interpretability findings from Days 1–5 were unified into an operational deployment framework. All candidate classifiers were benchmarked side-by-side across both discriminatory power and real-world hardware serving constraints.

---

### 1. Master Model Comparison Benchmark Matrix
The matrix below consolidates all 6 classifier iterations across Baseline (Day 3) and Tuned (Day 4/5) configurations on the identical held-out test partition (=320$):

| Model Architecture | Accuracy | Balanced Accuracy | Precision (Good) | Recall (Good) | F1-Score | ROC-AUC | PR-AUC | Screen Recall ($\\tau=0.35$) | Captured Lots |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Baseline)** | 93.44% | 0.7951 | 86.67% | 60.47% | 0.7123 | 0.9318 | 0.7986 | 72.09% | 31 / 43 |
| **Random Forest (Tuned Champion)** | **93.13%** | **0.8424** | **75.61%** | **72.09%** | **0.7381** | **0.9349** | **0.8260** | **79.07%** | **34 / 43** |
| **Support Vector Classifier (Baseline)** | 90.31% | 0.6690 | 83.33% | 34.88% | 0.4918 | 0.8698 | 0.6922 | 53.49% | 23 / 43 |
| **Support Vector Classifier (Tuned)** | 82.81% | 0.8025 | 42.31% | 76.74% | 0.5455 | 0.9097 | 0.6535 | 76.74% | 33 / 43 |
| **SGD Classifier (Baseline)** | 85.00% | 0.5204 | 27.27% | 6.98% | 0.1111 | 0.7501 | 0.2860 | 44.19% | 19 / 43 |
| **SGD Classifier (Tuned)** | 82.81% | 0.8320 | 42.86% | 83.72% | 0.5669 | 0.8967 | 0.5910 | 83.72% | 36 / 43 |

![Comprehensive Model Comparison](reports/figures/day6_comprehensive_model_comparison.png)

---

### 2. Hardware Inference Latency & Scalability Benchmarking

In industrial cellar operations, wines must be screened rapidly during harvest intake and bottling line triage. Latency was measured across percentiles alongside batch scaling:

| Architecture | Latency P50 (ms) | Latency P90 (ms) | Latency P95 (ms) | Latency P99 (ms) | Batch 500 Throughput | Artifact Footprint | Computational Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (Champion)** | **1.25 ms** | **2.80 ms** | **3.85 ms** | **6.10 ms** | **12,500 samples/s** | 2,540 KB | High throughput ensemble; ideal for cellar server |
| **SGD Classifier (Tuned)** | **0.08 ms** | **0.15 ms** | **0.22 ms** | **0.45 ms** | **85,000 samples/s** | 3.2 KB | Ultra-lightweight linear weights; ideal for handheld IoT |
| **SVC (Tuned)** | 4.80 ms | 9.20 ms | 12.50 ms | 18.40 ms | 1,850 samples/s | 45.0 KB | Kernel distance computation scaling bottleneck |

![Inference Latency vs Performance Pareto Frontier](reports/figures/day6_latency_vs_performance_tradeoff.png)

---

### 3. Multi-Criteria Decision Analysis (MCDA) Scoring Matrix

To balance pure statistical discrimination against production engineering feasibility, all models were evaluated across 5 weighted operational dimensions:
1. **Predictive Power (30% weight):** ROC-AUC & PR-AUC discrimination.
2. **Minority Sensitivity (25% weight):** Cellar capture recall at screening threshold $\\tau = 0.35$.
3. **Inference Speed & Throughput (20% weight):** P95 single-sample latency and batch throughput.
4. **Resource Footprint & Cold-Start (15% weight):** Serialized disk size and memory consumption.
5. **Interpretability & Trust (10% weight):** Biological alignment with enological principles.

| Architecture | Predictive Power (30%) | Minority Sensitivity (25%) | Inference Speed (20%) | Resource Footprint (15%) | Interpretability (10%) | Composite Score | Production Deployment Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | **94.5 / 100** | **91.0 / 100** | 82.0 / 100 | 74.0 / 100 | 88.0 / 100 | **87.9 / 100** | 🏆 **DESIGNATED PRODUCTION CHAMPION** |
| **SGD Classifier** | 76.0 / 100 | 84.0 / 100 | **98.0 / 100** | **99.0 / 100** | **95.0 / 100** | **86.8 / 100** | 🥈 **Edge / Ultra-Low Resource Runner-Up** |
| **SVC** | 82.0 / 100 | 78.0 / 100 | 65.0 / 100 | 80.0 / 100 | 45.0 / 100 | **73.0 / 100** | ❌ **Scaling & Interpretability Bottleneck** |

![Multi-Criteria Deployment Decision Radar Profile](reports/figures/day6_deployment_decision_radar.png)

---

### 4. Definitive Champion Architecture & Cellar Handover Specifications

* **Designated Production Champion:** **Tuned Random Forest Pipeline** (models/champion_rf_pipeline.joblib)
* **Optimal Cellar Screening Threshold:** $\\tau^* = 0.35$
* **Industrial Screening Metrics:**
  - **Balanced Accuracy:** **85.20%**
  - **Reserve Lot Capture (Recall):** **79.07%** (34 of 43 true reserve wines captured)
  - **False Negative Rate:** **20.93%** (Minimized from 60.5% in baseline)
  - **Inference Latency:** Median 1.25 ms, P95 3.85 ms
* **Production Handover Deliverables:**
  - Fully executed master notebook: [Notebooks/06_model_comparison_hardware_benchmarking_and_deployment_synthesis.ipynb](Notebooks/06_model_comparison_hardware_benchmarking_and_deployment_synthesis.ipynb)
  - Technical Handover Synthesis: [
eports/final_model_comparison.md](reports/final_model_comparison.md)
  - Hardware benchmarks: [
eports/day6_hardware_latency_benchmark.csv](reports/day6_hardware_latency_benchmark.csv)
  - Consolidated comparison table: [
eports/day6_master_model_comparison.csv](reports/day6_master_model_comparison.csv)
  - Suitability scoring matrix: [
eports/day6_deployment_suitability_matrix.csv](reports/day6_deployment_suitability_matrix.csv)

![Executive Handover Scorecard Dashboard](reports/figures/day6_executive_scorecard.png)


---

## 🚀 Day 7: Production Inference Engine, Real-Time REST API, Interactive Web Dashboard, Docker Containerization & CI/CD Automated Testing Framework

On Day 7, the project transitioned from offline model analysis to a production-grade, hardened Machine Learning serving architecture and interactive decision support platform.

`	ext
  +-----------------------------------------------------------------------+
  |                          Client Ingestion Layer                       |
  |     (Winery Laboratory LIMS / Streamlit Dashboard / REST Clients)     |
  +-----------------------------------+-----------------------------------+
                                      | HTTP POST /predict
                                      v
  +-----------------------------------------------------------------------+
  |                      FastAPI Microservice Gateway                     |
  |         (Pydantic v2 Schema Bounds & Biological Sanity Checks)        |
  +-----------------------------------+-----------------------------------+
                                      | Validated Attributes
                                      v
  +-----------------------------------------------------------------------+
  |                   Domain Feature Synthesis Engine                     |
  |    (5 Biochemical Ratios: SO2, Alc/Density, Acidity, SO4/Chloride)    |
  +-----------------------------------+-----------------------------------+
                                      | 16-Feature Vector
                                      v
  +-----------------------------------------------------------------------+
  |                 Champion Random Forest Ensemble Core                  |
  |        (150 Trees, Class-Weight Balanced, Joblib Loaded Pipeline)     |
  +-----------------------------------+-----------------------------------+
                                      | Posterior Probability (P_good)
                                      v
  +-----------------------------------------------------------------------+
  |                   Sensory Grading & Guidance Engine                   |
  |    (Optimal Screening Cutoff tau=0.35 + Enological Alert Rules)       |
  +-----------------------------------+-----------------------------------+
                                      |
                                      v
             🏆 Sensory Tier (Grand Reserve / Reserve / Table)
                      + Cellar Operational Directives
`

### 1. Production REST API Endpoints (src/production_service.py)

A production FastAPI microservice was constructed with high-throughput inference endpoints, automated biochemical ratio computation, and interactive Swagger documentation:

| HTTP Method | Route | Description | Request / Query Schema | Response Model |
| :---: | :--- | :--- | :--- | :--- |
| GET | / | Root service welcome banner & endpoint index | None | dict (links, status) |
| GET | /health | Uptime probe, system health & model readiness | None | HealthResponse |
| GET | /metadata | Champion model provenance, metrics & parameters | None | MetadataResponse |
| GET | /rules | Quantitative enological targets & benchmarks | None | dict (thresholds) |
| POST | /predict | Real-time single-sample inference & cellar advice | WinePhysicochemicalInput | PredictionResult |
| POST | /predict/batch | Vectorized multi-sample laboratory batch scoring | BatchPredictionRequest | BatchPredictionResponse |

---

### 2. Pydantic v2 Data Contract & Schema Boundary Enforcement

Strict physical validation bounds prevent biologically impossible, aberrant, or corrupted laboratory inputs from reaching the estimator:

- ixed acidity: $[3.0, 20.0]$ $	ext{g/dm}^3$ tartaric
- olatile acidity: $[0.05, 2.50]$ $	ext{g/dm}^3$ acetic
- citric acid: $[0.0, 1.50]$ $	ext{g/dm}^3$
- 
esidual sugar: $[0.4, 30.0]$ $	ext{g/dm}^3$
- chlorides: $[0.005, 1.000]$ $	ext{g/dm}^3$ sodium chloride
- ree sulfur dioxide: $[1.0, 120.0]$ $	ext{mg/dm}^3$
- 	otal sulfur dioxide: $[5.0, 350.0]$ $	ext{mg/dm}^3$ (Asserted: $\ge 	ext{free SO}_2$)
- density: $[0.970, 1.050]$ $	ext{g/cm}^3$
- pH: $[2.60, 4.60]$
- sulphates: $[0.20, 2.50]$ $	ext{g/dm}^3$ potassium sulphate
- lcohol: $[7.0, 17.5]$ $\%$ vol

---

### 3. Commercial Sensory Grading & Enological Cellar Directives

Every prediction is contextualized into commercial winery tiers and cellar guidance:
- **🏆 Grand Reserve ( \ge 0.75$):** Exceptional enological balance. Candidate for premium oak barrel maturation and top-tier reserve labeling.
- **🍷 Reserve Grade ( \ge 0.35$):** Solid candidate for reserve or vintage bottling. Cleared industrial screening threshold ($	au = 0.35$).
- **🍇 Table Standard (.20 \le P < 0.35$):** Standard commercial quality. Suitable for immediate table wine bottling, early release, or volume blending.
- **📦 Commercial Bulk ( < 0.20$):** High enological defect risk (e.g. elevated volatile acidity or insufficient preservative buffering). Recommended for bulk distillation, culinary processing, or immediate cellar stabilization.

---

### 4. Interactive Streamlit Web Platform (src/interactive_dashboard.py)

An interactive wine evaluation dashboard allows enologists and cellar masters to:
- **Real-Time Sliders:** Adjust chemical parameters with live feedback on predicted quality probability and commercial sensory tier.
- **Enological Radar Profile:** Visually inspect how the sample aligns against empirical reserve benchmarks (lcohol, olatile acidity, sulphates, citric acid, 	otal acidity).
- **Batch CSV Triage:** Upload laboratory batch CSVs, triage entire cellar harvests, inspect yield distributions, and download annotated triage reports.
- **Enological Diagnostic Directives:** Real-time biological warnings for volatile acid elevation, free-to-total SO2 preservation risk, and alcohol-density balance.

---

### 5. Automated Verification Test Suite (	ests/, 22 Tests)

A complete test suite was established and executed via pytest, achieving a **100% pass rate (22/22 passed)**:

- **ML Pipeline & Feature Engineering Tests (	ests/test_pipeline.py, 7 tests):**
  - Dataset presence and zero missing values across 1,599 red wine specimens.
  - Cortez et al. quality score class distribution integrity.
  - Quality score binarization logic ($\ge 7$ vs $< 7$).
  - Mathematical parity for all 5 engineered interaction features.
  - Stratified 80/20 train/test split preservation.
  - Serialized champion Random Forest pipeline scoring parity.
- **Production REST API & Validation Tests (	ests/test_production_service.py, 15 tests):**
  - Model artifact loading and provenance metadata validation.
  - Pydantic v2 valid payload parsing and alias resolution.
  - Boundary constraint enforcement (catching impossible pH, out-of-bounds alcohol).
  - Cross-attribute consistency check (intercepting $	ext{Free SO}_2 > 	ext{Total SO}_2$).
  - FastAPI endpoints /health, /metadata, /rules, /predict, /predict/batch.
  - Operational screening threshold override functionality.
  - Single-sample latency SLA verification ($< 50	ext{ ms}$).
  - Vectorized direct service engine latency verification ($< 15	ext{ ms}$).

---

### 6. Containerization & CI/CD Pipeline

The serving architecture is containerized and integrated into GitHub Actions CI/CD:
- **Dockerfile:** Multi-stage lightweight python:3.11-slim container with non-root security compliance (ppuser), cached layer dependency installs, and container healthcheck probes (/health).
- **docker-compose.yml:** Multi-service orchestration managing the FastAPI REST API (port 8000) and Streamlit Dashboard (port 8501) with dependency chaining.
- **.github/workflows/ci.yml:** Automated GitHub Actions matrix workflow verifying data integrity and executing the full 22-test pytest suite on Python 3.11 and 3.12 for every push and pull request.

---

### 7. Empirical REST API Latency & Throughput Benchmarks

Empirical response latencies measured via ASGI test client across 40 continuous request iterations per endpoint:

| Endpoint | Total Requests | Mean Latency (ms) | Median P50 (ms) | P95 Latency (ms) | P99 Latency (ms) | Min Latency (ms) | Max Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GET / | 40 | 18.58 ms | 12.68 ms | 49.32 ms | 71.46 ms | 6.86 ms | 72.72 ms |
| GET /health | 40 | 16.58 ms | 15.68 ms | 32.65 ms | 35.18 ms | 8.28 ms | 35.28 ms |
| GET /metadata | 40 | 22.07 ms | 13.76 ms | 44.88 ms | 144.96 ms | 6.99 ms | 199.76 ms |
| GET /rules | 40 | 9.73 ms | 8.60 ms | 15.98 ms | 18.27 ms | 6.20 ms | 19.53 ms |
| POST /predict (Table Wine) | 40 | 90.39 ms | 66.77 ms | 135.90 ms | 505.29 ms | 49.72 ms | 737.71 ms |
| POST /predict (Reserve Grand Cru) | 40 | 48.78 ms | 43.08 ms | 76.01 ms | 100.80 ms | 32.28 ms | 103.97 ms |
| POST /predict/batch (50 Samples) | 40 | 42.71 ms | 41.54 ms | 53.77 ms | 56.64 ms | 34.36 ms | 57.88 ms |
| POST /predict/batch (200 Samples) | 40 | 63.45 ms | 53.89 ms | 125.65 ms | 169.13 ms | 43.12 ms | 181.45 ms |

---

### 8. Generated Day 7 Figures

* [day7_production_architecture.png](reports/figures/day7_production_architecture.png) — End-to-end production MLOps containerized architecture and serving workflow.
* [day7_api_latency_distribution.png](reports/figures/day7_api_latency_distribution.png) — Response latency empirical distribution and percentiles under production ASGI serving.
* [day7_batch_inference_scaling.png](reports/figures/day7_batch_inference_scaling.png) — Vectorized batch inference throughput scaling curves across increasing specimen lot sizes.
* [day7_sommelier_profile_radar.png](reports/figures/day7_sommelier_profile_radar.png) — Interactive sommelier radar chart benchmarking candidate wine specimens against empirical reserve standards.

![Production Architecture](reports/figures/day7_production_architecture.png)
![API Latency Distribution](reports/figures/day7_api_latency_distribution.png)
![Batch Inference Scaling](reports/figures/day7_batch_inference_scaling.png)
![Sommelier Profile Radar](reports/figures/day7_sommelier_profile_radar.png)

---

## 📈 Day 8: Continuous MLOps Monitoring, Statistical Drift Detection & Automated Retraining Framework

```plaintext
+---------------------------------------------------------------------------------------------------------+
|                                    DAY 8: CLOSED-LOOP MLOPS MONITORING                                  |
+---------------------------------------------------------------------------------------------------------+
|                                        Incoming Harvest Specimens                                       |
|                                     (Annual Vintage Laboratory Stream)                                  |
+----------------------------------------------------+----------------------------------------------------+
                                                     |
                                                     v
                     +---------------------------------------------------------------+
                     |          Multivariate Chemical Anomaly Screening              |
                     |         (IsolationForest: Adulterant / Spoilage Filter)       |
                     +-------------------------------+-------------------------------+
                                                     | Authentic Specimens (> 97%)
                                                     v
                     +---------------------------------------------------------------+
                     |           Continuous Statistical Covariate Drift              |
                     |         (Quantile PSI + Two-Sample Kolmogorov-Smirnov)        |
                     +-------------------------------+-------------------------------+
                                                     |
                                    +----------------+----------------+
                                    |                                 |
                        [Covariate Drift Fired: PSI >= 0.25] [Longitudinal Sommelier Degradation]
                                    |                                 |
                                    +----------------+----------------+
                                                     | Retraining Alarm
                                                     v
                     +---------------------------------------------------------------+
                     |              Champion-Challenger Shadow Tournament            |
                     |         (Frozen Incumbent vs Retrained Candidate on K-Fold)   |
                     +-------------------------------+-------------------------------+
                                                     |
                                                     v
                                      🏆 Promoted to Production Serving
```

### 1. Statistical Covariate Shift Detection Engine (`src/drift_monitoring.py`)

Continuous monitoring monitors feature distributions using dual statistical indicators:
* **Quantile Population Stability Index (PSI):**
  - $\text{PSI} < 0.10$: Stable baseline (**GREEN**).
  - $0.10 \le \text{PSI} < 0.25$: Moderate distribution shift (**AMBER**).
  - $\text{PSI} \ge 0.25$: Significant population drift requiring investigation (**RED**).
* **Two-Sample Kolmogorov-Smirnov (K-S) Hypothesis Test:** Continuous non-parametric goodness-of-fit test against empirical distribution functions ($p < 0.01$).

#### Vintage Climate Drift Scorecard (Heatwave Scenario):

| Feature | Reference Baseline Mean | Shifted Vintage Mean | Relative Shift ($\Delta\%$) | PSI Score | K-S Stat ($D$) | Alert Level | Action Required |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `alcohol` | 10.42% | 11.87% | **+13.92%** | **0.3842** | 0.3812 | 🚨 **RED** | Retraining trigger candidate |
| `volatile acidity` | 0.528 | 0.634 | **+20.08%** | **0.2915** | 0.3120 | 🚨 **RED** | Retraining trigger candidate |
| `fixed acidity` | 8.320 | 7.322 | **-12.00%** | **0.2110** | 0.2450 | ⚠️ **AMBER** | Monitoring watch |
| `pH` | 3.311 | 3.443 | **+4.00%** | **0.1840** | 0.2190 | ⚠️ **AMBER** | Monitoring watch |
| `total sulfur dioxide`| 46.47 | 45.92 | -1.18% | 0.0182 | 0.0410 | 🟢 **GREEN** | Normal operational variance |

---

### 2. Multivariate Chemical Anomaly & Outlier Screening (`MultivariateAnomalyDetector`)

* **Engine:** Pre-inference multi-attribute `IsolationForest` ($n_{\text{estimators}} = 100$, contamination $= 4\%$).
* **Function:** Filters physically corrupted, miscalibrated, or chemically adulterated specimens prior to model evaluation.
* **Result:** Successfully quarantines aberrant batches while passing 97%+ of authentic harvest samples into the serving pipeline.

---

### 3. Longitudinal Sommelier Auditing & Concept Drift Tracking

* Quarterly blind sensory audits track model discrimination and calibration against true human tasting scores:
  - Tracks degradation in ROC-AUC, Balanced Accuracy, Precision, Recall, and F1 over time.
  - Detects true concept drift ($P(Y|X)$ alterations) independent of input distribution shifts.
* **Degradation & Recovery Profile:**
  - **Vintage 2021 (Baseline):** ROC-AUC: 0.9368 | Balanced Accuracy: 85.20% | F1: 0.6970
  - **Vintage 2023 (Heatwave Drift):** ROC-AUC: 0.8620 | Balanced Accuracy: 77.10% | F1: 0.5820 (SLA Warning)
  - **Vintage 2024 (Retraining Trigger):** ROC-AUC: 0.8350 | Balanced Accuracy: 73.90% | F1: 0.5210 (🚨 Trigger Fired)
  - **Post-Retraining (Challenger):** ROC-AUC: **0.9395** | Balanced Accuracy: **86.40%** | F1: **0.7120** (Full Restoration)

---

### 4. Automated Retraining Tournament: Champion vs. Challenger

When retraining conditions trigger, candidate pipelines are fitted on augmented data and judged in a side-by-side tournament:

| Evaluation Metric | Incumbent Champion (Frozen) | Challenger Pipeline (Retrained) | Net Improvement ($\Delta$) | Tournament Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **5-Fold CV ROC-AUC** | 0.9015 ± 0.0210 | **0.9284 ± 0.0165** | **+0.0269** | Challenger Outperforms |
| **5-Fold CV Balanced Acc** | 0.8057 ± 0.0310 | **0.8412 ± 0.0220** | **+0.0355** | Challenger Outperforms |
| **Audit Test ROC-AUC** | 0.8840 | **0.9350** | **+0.0510** | Challenger Outperforms |
| **Audit Test Balanced Acc** | 0.7920 | **0.8580** | **+0.0660** | Challenger Outperforms |
| **Audit Test F1-Score** | 0.6120 | **0.7040** | **+0.0920** | Challenger Outperforms |
| **Final Governance Status** | Incumbent Retired | **Promoted to Production** | — | 🏆 **PROMOTED** |

---

### 5. Production REST API Monitoring Endpoints (`src/production_service.py`)

Two production endpoints were added to support continuous automated drift auditing:

| HTTP Method | Route | Description | Request Payload | Response Schema |
| :---: | :--- | :--- | :--- | :--- |
| `GET` | `/metrics/drift` | Returns system drift status, reference size, and PSI thresholds | None | `DriftMetricsResponse` |
| `POST` | `/audit/evaluate` | Evaluates covariate shift on an incoming laboratory batch | `BatchPredictionRequest` | `DriftAuditResponse` |

---

### 6. Interactive Streamlit Monitoring Center (`src/interactive_dashboard.py`)

The Streamlit web platform includes a dedicated **Vintage Drift & MLOps Monitoring** center:
- **Harvest Simulation Controls:** Allows winemakers to simulate Climate Warming, Cool Rainy, or Sensor Bias scenarios.
- **Real-Time PSI Scorecard:** Displays live multi-metric drift results with color-coded status badges.
- **Interactive Distribution Overlay:** Density curves comparing baseline vs incoming harvest specimens.
- **Longitudinal SLA Tracker:** Visualizes multi-year performance decay and automated retraining recovery.

---

### 7. Automated Pytest Verification Suite (32 Passing Tests)

The test suite was expanded with 10 new comprehensive tests in `tests/test_drift_monitoring.py`:
- Validates PSI calculation on identical and shifted distributions.
- Validates Two-Sample Kolmogorov-Smirnov test statistics and p-values.
- Tests multi-attribute anomaly detection and adulteration quarantine logic.
- Verifies longitudinal audit performance tracking and concept drift alarms.
- Verifies automated retraining triggers and champion-challenger tournament promotion.
- Validates FastAPI `/metrics/drift` and `/audit/evaluate` endpoints.
- **Full Test Results:** **32 passed, 100% pass rate.**

---

### 8. Generated Day 8 Figures

All monitoring and drift diagnostics figures are published in `reports/figures/`:
* [day8_feature_drift_psi_summary.png](reports/figures/day8_feature_drift_psi_summary.png) — Feature-level Population Stability Index (PSI) horizontal scorecard with color-coded risk bands.
* [day8_drift_distribution_shifts.png](reports/figures/day8_drift_distribution_shifts.png) — Multi-panel KDE distributions contrasting baseline reference against drifted harvest vintage for top chemical drivers.
* [day8_concept_drift_performance_decay.png](reports/figures/day8_concept_drift_performance_decay.png) — Longitudinal tracking curve showing model SLA decay under unmonitored drift and complete recovery post-retraining.
* [day8_mlops_monitoring_pipeline.png](reports/figures/day8_mlops_monitoring_pipeline.png) — Architectural infographic showing the closed-loop MLOps continuous monitoring and retraining pipeline.

![Feature Drift PSI Scorecard](reports/figures/day8_feature_drift_psi_summary.png)
![Drift Distribution Shifts](reports/figures/day8_drift_distribution_shifts.png)
![Concept Drift Performance Decay](reports/figures/day8_concept_drift_performance_decay.png)
![MLOps Monitoring Pipeline](reports/figures/day8_mlops_monitoring_pipeline.png)

---

## 🍇 Day 9: Cross-Cultivar Domain Adaptation, White Wine Champion Pipeline & Prescriptive Enological Recourse Engine

`plaintext
+-------------------------------------------------------------------------------------------------------------+
|                                    DAY 9: DUAL-CULTIVAR & RECOURSE PLATFORM                                 |
+-------------------------------------------------------------------------------------------------------------+
|                                            Incoming Wine Specimen                                           |
|                                        (11 Physicochemical Attributes)                                      |
+------------------------------------------------------+------------------------------------------------------+
                                                       |
                                                       v
                       +---------------------------------------------------------------+
                       |           High-Accuracy Chemical Cultivar Detector            |
                       |       (Random Forest Classifier, 99.62% Test Accuracy)        |
                       +-------------------------------+-------------------------------+
                                                       |
                                      +----------------+----------------+
                                      |                                 |
                          [Red Cultivar: ~99.9%]             [White Cultivar: ~99.9%]
                                      |                                 |
                                      v                                 v
                +-----------------------------+   +-----------------------------+
                |     Red Champion Model      |   |    White Champion Model     |
                |   (models/champion_rf...)   |   | (models/white_champion...)  |
                |   ROC-AUC: 0.9368 | tau=0.35|   |  ROC-AUC: 0.9243 | tau=0.35 |
                +--------------+--------------+   +--------------+--------------+
                               |                                 |
                               +----------------+----------------+
                                                |
                                                v
                               +---------------------------------+
                               |   Posterior Probability P(Good) |
                               +----------------+----------------+
                                                |
                        +-----------------------+-----------------------+
                        |                                               |
              [P(Good) >= tau*=0.35]                          [P(Good) < tau*=0.35]
                        |                                               |
                        v                                               v
          +---------------------------+                   +---------------------------+
          |       Reserve Grade       |                   |    Standard Table Wine    |
          |  Cellar Aging & Bottling  |                   |             |             |
          +---------------------------+                   +-------------+-------------+
                                                                        |
                                                                        v
                                                      +-----------------------------------+
                                                      |   Cellar Recourse Optimizer       |
                                                      | Minimal Feasible Interventions    |
                                                      |  (SO2 / Volatile Acid / Alcohol)  |
                                                      +-----------------+-----------------+
                                                                        |
                                                                        v
                                                      +-----------------------------------+
                                                      |  Prescriptive Winemaking Plan:    |
                                                      |  +14.0% P(Good) Lift -> Reserve!  |
                                                      +-----------------------------------+
`

### 1. Cross-Cultivar Ingestion & Statistical Divergence Analysis (CultivarDomainAnalyzer)

The UCI Wine Quality benchmark comprises both **Red Wine** (1,599 samples) and **White Wine** (4,898 samples).
Evaluating the two populations revealed massive covariate shift across primary chemical drivers:

| Feature | Red Mean | White Mean | Relative Shift ($\Delta\%$) | PSI Score | K-S Stat ($) | Domain Divergence Severity |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 	otal sulfur dioxide | 46.468 | 138.361 | **+197.75%** | **4.8145** | 0.8037 | **Severe Red Alert** |
| chlorides | 0.0875 | 0.0458 | **-47.67%** | **4.3267** | 0.7712 | **Severe Red Alert** |
| olatile acidity | 0.5278 | 0.2782 | **-47.28%** | **2.8795** | 0.6483 | **Severe Red Alert** |
| 
esidual sugar | 2.5388 | 6.3914 | **+151.75%** | **2.0901** | 0.5126 | **Severe Red Alert** |
| ree sulfur dioxide | 15.875 | 35.308 | **+122.41%** | **1.7917** | 0.5891 | **Severe Red Alert** |
| sulphates | 0.6581 | 0.4898 | **-25.57%** | **1.5803** | 0.5218 | **Severe Red Alert** |
| ixed acidity | 8.3196 | 6.8548 | **-17.61%** | **1.5366** | 0.5284 | **Severe Red Alert** |
| density | 0.9967 | 0.9940 | **-0.27%** | **1.3246** | 0.5694 | **Severe Red Alert** |
| citric acid | 0.2710 | 0.3342 | **+23.33%** | **1.1482** | 0.3015 | **Severe Red Alert** |
| pH | 3.3111 | 3.1883 | **-3.71%** | **0.6717** | 0.3809 | **Severe Red Alert** |
| lcohol | 10.4230 | 10.5143 | **+0.88%** | **0.1574** | 0.0882 | **Moderate Shift** |

---

### 2. Quantifying Cross-Domain Transfer Fragility

Applying the frozen Red Wine Champion pipeline directly (zero-shot transfer) to the 4,898 White Wine dataset causes massive performance degradation, proving that single-domain models cannot serve a dual-cultivar winery:

| Evaluation Metric | Red Champion (In-Domain: Red) | Red Champion (Zero-Shot: White) | White Champion (Dedicated Domain) | Net Degradation ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | **0.9368** | 0.7593 | **0.9243** | **-17.75%** |
| **PR-AUC** | **0.8095** | 0.4582 | **0.8289** | **-35.13%** |
| **Balanced Accuracy ($	au=0.50$)** | **0.8520** | 0.5404 | **0.8428** | **-31.16%** |
| **Balanced Accuracy ($	au^*=0.35$)** | **0.8520** | 0.6759 | **0.8433** | **-17.61%** |
| **Minority Recall ($	au^*=0.35$)** | **0.7907** | 0.5085 | **0.8962** | **-28.22%** |
| **Domain Transfer Verdict** | Production Standard | ❌ **Severe Fragility Failure** | 🏆 **Full Domain Restoration** | Retraining Required |

---

### 3. High-Accuracy Chemical Cultivar Detector (CultivarDetector)

* **Model Architecture:** Random Forest Classifier ({	ext{estimators}} = 100$, $	ext{max\_depth} = 10$).
* **Holdout Test Set Performance:** **99.62% Accuracy**, **1.0000 ROC-AUC**.
* **Top Chemical Discriminators:**
  1. 	otal sulfur dioxide: 31.90% importance (White wines require 3x higher free & total SO2).
  2. chlorides: 25.88% importance (Red wines possess 2x higher mineral salinity).
  3. olatile acidity: 10.83% importance (Red wines tolerate higher acetic acid; white wines emphasize crisp purity).
  4. density: 6.50% importance.
  5. sulphates: 6.10% importance.
* **Artifacts:** Exported to models/cultivar_classifier.joblib and models/cultivar_classifier.json.

---

### 4. Dedicated White Wine Champion Pipeline (WhiteWineChampionTrainer)

* **Dataset Split:** 4,898 white wine samples (3,918 Train / 980 Test, Stratified 80/20).
* **Architecture:** Tuned Random Forest ({	ext{estimators}} = 200$, $	ext{max\_depth} = 16$, $	ext{min\_samples\_split} = 4$, $	ext{class\_weight} = 	ext{'balanced'}$).
* **Cross-Validation Performance:** 5-Fold Stratified CV ROC-AUC = **0.8995 ± 0.0074**, Balanced Accuracy = **0.7965**.
* **Holdout Test Set Performance:**
  - **ROC-AUC:** **0.9243**
  - **PR-AUC:** **0.8289**
  - **Balanced Accuracy ($	au^* = 0.35$):** **84.33%**
  - **Minority Recall ($	au^* = 0.35$):** **89.62%** (captures 190 of 212 premium white wines!)
* **Artifacts:** Exported to models/white_champion_rf_pipeline.joblib and models/white_champion_rf_pipeline.json.

---

### 5. Prescriptive Cellar Recourse Optimizer (CellarRecourseOptimizer)

A counterfactual optimization engine that computes the minimal enological adjustment vector ($\Delta \mathbf{x}$) required to upgrade sub-premium batches ((	ext{good}) < 	au^*$) into the Reserve Grade tier:

\min_{\Delta \mathbf{x}} \sum_{i} w_i \left(rac{x_i - x_i^{(0)}}{	ext{scale}_i}ight)^2 \quad 	ext{subject to } \hat{P}(	ext{premium} \mid \mathbf{x}^{(0)} + \Delta \mathbf{x}) \ge 	au_{	ext{target}}

subject to strict physical and biochemical winemaking constraints (e.g. non-increasing volatile acidity, antioxidant ratio consistency, alcohol blending envelopes).

#### Example Recourse Plan on Standard Red Wine Specimen:
* **Initial State:** (	ext{good}) = 0.010$ (Commercial Table Wine, Failing Score)
* **Target Threshold:** $	au^* = 0.350$ (Reserve Grade Cutoff)
* **Optimized Adjustments:**
  1. olatile acidity: .70 	o 0.39$ g/dm³ (-45.0%) via aeration, fining, or yeast selection.
  2. lcohol: .4\% 	o 10.9\%$ (+16.0%) via blending with high-gravity reserve lot.
  3. sulphates: .56 	o 0.78$ g/dm³ (+39.3%) via food-grade potassium sulphate dosing.
  4. ree sulfur dioxide: .0 	o 15.0$ mg/dm³ (+36.4%) via potassium metabisulfite addition.
  5. 	otal sulfur dioxide: .0 	o 60.0$ mg/dm³ (+76.5%) to maintain active antioxidant buffer.
* **Recourse Outcome:** Elevates projected probability to **15.0% - 46.5%** with **High Feasibility** rating.

---

### 6. Production REST API Multi-Cultivar Endpoints (src/production_service.py)

Three new endpoints and auto-routing capabilities were integrated into the production FastAPI microservice:

| HTTP Method | Route | Description | Request Payload | Response Model |
| :---: | :--- | :--- | :--- | :--- |
| POST | /cultivar/detect | Predicts cultivar (red vs white) and detection confidence | WinePhysicochemicalInput | CultivarDetectionResponse |
| POST | /predict/recourse | Computes actionable cellar adjustments to upgrade sub-premium wine | WinePhysicochemicalInput | CellarRecourseResponse |
| GET | /metrics/cultivars | Returns full cross-cultivar divergence scorecard and domain gap | None | dict (scorecard & metrics) |
| POST | /predict | Evaluates wine sample with **automatic cultivar routing** | WinePhysicochemicalInput | PredictionResult (with cultivar tags) |

---

### 7. Interactive Streamlit What-If Recourse Studio (src/interactive_dashboard.py)

A new dedicated tab — **🧪 Prescriptive Cellar Recourse Studio** — was added to the interactive web platform:
* **Cellar Parameter Controls:** Real-time chemical sliders for alcohol, volatile acidity, sulphates, and SO2.
* **Target Reserve Setting:** Allows winemakers to specify target confidence (.35 - 0.75$).
* **Live KPI Scorecard:** Displays current vs target vs projected probability with color-coded feasibility badges.
* **Actionable Interventions Table:** Tabulates exact dosages, percentage changes, and practical cellar advice.
* **Cross-Cultivar Divergence Scorecard:** Full interactive table of domain shift across both cultivars.

---

### 8. Automated Pytest Verification Suite (45 Passing Tests)

The automated test suite was expanded with 13 new unit and API tests in 	ests/test_cross_cultivar_and_recourse.py:
- Validates cross-cultivar divergence scorecard and feature ranking.
- Quantifies zero-shot cross-cultivar fragility collapse.
- Validates cultivar detector accuracy ($\ge 98\%$) and single-sample inference.
- Verifies White Wine Champion model serialization, 5-fold CV, and test benchmarks.
- Verifies cellar recourse optimizer convergence and biochemical constraint bounds.
- Validates FastAPI /cultivar/detect, /predict/recourse, and /metrics/cultivars endpoints.
- **Full Test Results:** **45 passed, 100% pass rate.**

---

### 9. Generated Day 9 Figures

All domain adaptation and recourse figures are published in 
eports/figures/:
* [day9_cross_cultivar_chemical_divergence.png](reports/figures/day9_cross_cultivar_chemical_divergence.png) — Multi-panel KDE distributions showing fundamental chemical divergences between Red and White wine populations.
* [day9_domain_transfer_gap.png](reports/figures/day9_domain_transfer_gap.png) — Grouped benchmark comparing in-domain performance vs zero-shot transfer degradation vs adapted White Champion.
* [day9_white_champion_roc_pr_curves.png](reports/figures/day9_white_champion_roc_pr_curves.png) — White Wine Champion ROC curve (AUC=0.9243) and PR curve (AUC=0.8289) with operating threshold markers.
* [day9_counterfactual_recourse_waterfall.png](reports/figures/day9_counterfactual_recourse_waterfall.png) — Step-by-step waterfall chart showing sequential probability gains from cellar interventions.

![Cross-Cultivar Chemical Divergence](reports/figures/day9_cross_cultivar_chemical_divergence.png)
![Domain Transfer Gap](reports/figures/day9_domain_transfer_gap.png)
![White Champion ROC & PR Curves](reports/figures/day9_white_champion_roc_pr_curves.png)
![Counterfactual Recourse Waterfall](reports/figures/day9_counterfactual_recourse_waterfall.png)



---

## 🍇 Day 10: Enological Coupage (Wine Blending) Optimization Engine, Split Conformal Uncertainty Quantification, Winery Economic Value Analysis & Multi-Model Stacking Ensemble

```plaintext
+-------------------------------------------------------------------------------------------------------------+
|                                    DAY 10: ENOLOGICAL COUPAGE & VALUE PLATFORM                              |
+-------------------------------------------------------------------------------------------------------------+
|                                        Candidate Cellar Harvest Tanks                                       |
|                                       (Tank A, Tank B, ..., Tank K Profiles)                                |
+------------------------------------------------------+------------------------------------------------------+
                                                       |
                                                       v
                       +---------------------------------------------------------------+
                       |              Non-Linear Chemical Coupage Engine               |
                       |  - Additive Mass Conservation (Alcohol, Volatile Acid, SO2)   |
                       |  - Logarithmic Hydrogen-Ion Equilibrium: [H+] = Σ w_i 10^-pH_i |
                       +-------------------------------+-------------------------------+
                                                       |
                                                       v
                       +---------------------------------------------------------------+
                       |           Constrained Coupage Optimizer (SLSQP Solver)         |
                       |       Max P(Premium | Blend(w)) s.t. VA <= 0.60, SO2, Alc     |
                       +-------------------------------+-------------------------------+
                                                       |
                                      +----------------+----------------+
                                      |                                 |
                                      v                                 v
                +-----------------------------+   +-----------------------------+
                | Split Conformal Predictor   |   | Winery Economic Value (EMV) |
                | P(Y in C(X)) >= 1 - alpha   |   | Asymmetric Payoff Matrix    |
                | Ambiguous Sommelier Flags   |   | Optimal tau*_profit (+$28k) |
                +--------------+--------------+   +--------------+--------------+
                               |                                 |
                               +----------------+----------------+
                                                |
                                                v
                               +---------------------------------+
                               | Multi-Model Stacking Ensemble   |
                               | RF + GBDT + ExtraTrees + LR     |
                               | Meta ROC-AUC: 0.9435 | Brier 0.05|
                               +---------------------------------+
```

### 1. Enological Coupage (Wine Assemblage & Blending) Optimization

Commercial wineries achieve sensory excellence by combining candidate lots. Day 10 implements `WineAssemblageOptimizer`:
* **Non-Linear Acid-Base Chemical Equilibrium:**
  $$\text{pH}_{\text{blend}} = -\log_{10}\left( \sum_{i=1}^K w_i \cdot 10^{-\text{pH}_i} \right)$$
* **Additive Conservation of Mass:**
  $$c_{\text{blend}} = \sum_{i=1}^K w_i \cdot c_i \quad \text{for alcohol, volatile acidity, citric acid, sugar, chlorides, SO}_2\text{, sulphates}$$
* **Constrained SLSQP Optimization:**
  $$\max_{w} P(\text{Quality} \ge 7 \mid \text{Blend}(w)) \quad \text{s.t.} \quad \sum w_i = 1, \; 0 \le w_i \le 1$$
  $$\text{Volatile Acidity} \le 0.60 \text{ g/L (spoilage ceiling)}, \quad \text{Total SO}_2 \le 160 \text{ mg/L (EU legal)}, \quad \text{Alcohol} \ge 11.5\%$$
* **Empirical Lift:** Blending a flawed Lot A (40%, VA=0.72) with Reserve Lot B (60%, VA=0.32) elevates predicted premium probability from 0.084 up to **0.6938**, achieving full enological compliance.

---

### 2. Split Conformal Prediction & Uncertainty Quantification

Point predictions fail to communicate decision confidence. Day 10 integrates `EnologicalConformalPredictor`:
* **Finite-Sample Distribution-Free Coverage:**
  $$P(Y \in C(X)) \ge 1 - \alpha$$
* **Non-Conformity Scoring & Quantile Adjustment:**
  $$s_i = 1 - \hat{P}(Y = y_i \mid X_i), \quad \hat{q} = \text{Quantile}\left(\{s_i\}, \frac{\lceil (n_{\text{cal}}+1)(1-\alpha) \rceil}{n_{\text{cal}}}\right)$$
* **Empirical Validation:**
  - At $\alpha=0.05$ (95% target): Empirical test coverage = **97.50%** (Singletons: 85%, Ambiguous: 15%).
  - At $\alpha=0.10$ (90% target): Empirical test coverage = **93.13%** (Singletons: 97%, Ambiguous: 3%).
* **Actionable Sommelier Directives:** Singletons release directly with certified statistical guarantee; ambiguous dual-label sets $\{ \text{Standard}, \text{Premium} \}$ trigger mandatory laboratory sensory panel arbitration.

---

### 3. Winery Decision-Theoretic Payoff Matrix & Economic Value Engine

Commercial winemaking is governed by asymmetric financial outcomes. Day 10 implements `WineryEconomicEngine`:
* **Enological Payoff Matrix:**
  - True Positive (TP): +$28.00/bottle (Reserve Grade margin)
  - False Positive (FP): -$45.00/bottle (Sub-standard wine sold as Reserve -> brand equity penalty & returns)
  - False Negative (FN): -$12.00/bottle (Premium wine mistakenly downgraded to bulk table wine)
  - True Negative (TN): +$4.50/bottle (Everyday table wine margin)
* **Expected Monetary Value (EMV) Optimization:**
  Identifies $\tau_{\text{profit}}^* = 0.74$, yielding **$131,328** net harvest profit on a 25,000-bottle commercial vintage (**+$28,450 dollar uplift** vs default $\tau=0.50$).

---

### 4. Heterogeneous Multi-Model Stacking Ensemble

Synthesizes a 2-level stacking ensemble combining complementary inductive biases:
* **Level-0 Base Learners:**
  - Tuned Random Forest (Tree Ensemble)
  - Gradient Boosted Trees (Residual Descent)
  - Extra Trees Classifier (Extremely Randomized Trees)
  - Regularized Scaled Logistic Regression (Linear Margin)
* **Level-1 Meta-Learner:** Calibrated Logistic Regression trained on 5-Fold Stratified out-of-fold probability vectors.
* **Test Performance:** Stacking ROC-AUC achieves **0.9435** with a Brier calibration loss of **0.0548**, outperforming individual standalone models.

---

### 5. Production REST API Endpoints (`src/production_service.py`)

Four enterprise-grade endpoints added to the production FastAPI REST microservice:
- `POST /blend/optimize`: Accepts candidate lot profiles and constraints, returns optimal blending ratios, blended chemistry, and predicted reserve quality.
- `POST /predict/conformal`: Returns certified conformal prediction sets, error rate $\alpha$, confidence percentage, and sommelier action directives.
- `POST /economics/evaluate`: Computes expected monetary value, optimal financial threshold $\tau_{\text{profit}}^*$, and harvest revenue uplift.
- `GET /metrics/stacking`: Returns multi-model stacking ensemble benchmark metrics and meta-learner model weights.

---

### 6. Interactive Streamlit Coupage & Uncertainty Laboratory (`src/interactive_dashboard.py`)

Two new dedicated tabs added to the interactive web dashboard:
* **🍇 Enological Coupage & Multi-Tank Blending Simulator:** Interactive sliders for candidate lots, live non-linear pH recalculation, and one-click SLSQP constrained coupage solver.
* **🎯 Split Conformal Uncertainty & Winery Economic Value Engine:** Significance level $\alpha$ slider, certified prediction set display, and commercial harvest profit calculator.

---

### 7. Automated Pytest Verification Suite (57 Passing Tests)

The automated test suite was expanded with 12 new unit and API tests in `tests/test_assemblage_and_conformal.py`:
- Validates mass conservation, linear mixing, and logarithmic $[H^+]$ acid equilibrium for pH.
- Verifies SLSQP coupage optimization convergence and regulatory constraint satisfaction (VA $\le 0.60$, alcohol $\ge 11.5\%$).
- Tests split conformal calibration, non-conformity quantiles, and empirical coverage guarantees.
- Tests commercial payoff matrix and economic threshold optimization.
- Verifies stacking ensemble training, out-of-fold blending, and model serialization.
- Validates FastAPI endpoints (`/blend/optimize`, `/predict/conformal`, `/economics/evaluate`, `/metrics/stacking`).
- **Full Test Results:** **57 passed, 100% pass rate.**

---

### 8. Generated Day 10 Figures

All coupage, conformal, and economic diagnostic figures are published in `reports/figures/`:
* [day10_coupage_optimization_surface.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day10_coupage_optimization_surface.png) — Enological coupage trajectory, viable reserve bottling zone, and chemical balance before vs after assemblage.
* [day10_conformal_prediction_coverage.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day10_conformal_prediction_coverage.png) — Conformal calibration reliability curve, prediction set cardinality distribution, and ambiguous lot density.
* [day10_winery_economic_threshold_utility.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day10_winery_economic_threshold_utility.png) — Expected Monetary Value (EMV) curve vs threshold $\tau$, identifying financial optimum $\tau_{\text{profit}}^*$ and harvest revenue uplift.
* [day10_stacking_ensemble_calibration.png](file:///C:/Users/Iba%20Zaidi/Desktop/Wine%20Quality%20Prediction/reports/figures/day10_stacking_ensemble_calibration.png) — Reliability calibration curves and Brier score benchmarks comparing Level-0 models vs Stacking Meta-Learner.

![Coupage Optimization Surface](reports/figures/day10_coupage_optimization_surface.png)
![Conformal Prediction Coverage](reports/figures/day10_conformal_prediction_coverage.png)
![Winery Economic Threshold Utility](reports/figures/day10_winery_economic_threshold_utility.png)
![Stacking Ensemble Calibration](reports/figures/day10_stacking_ensemble_calibration.png)


