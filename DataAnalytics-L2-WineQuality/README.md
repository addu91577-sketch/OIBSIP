# 🍷 Wine Quality Prediction — Physicochemical Classification
**Oasis Infobyte Data Analytics Internship — Task Level 2 (Task 2)**

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
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
- [ ] **Feature engineering:** consider binning quality scores into binary (good/bad) or 3-class (low/medium/high) groups and justify the decision
- [ ] **Train/test split with stratification** to preserve class ratios
- [ ] **Train 3 classifiers:** Random Forest, Stochastic Gradient Descent (SGD), and Support Vector Classifier (SVC)
- [ ] **Evaluate each:** accuracy, classification report, confusion matrix
- [ ] **Feature importance chart** for the Random Forest model
- [ ] **Comparison table:** summarise the performance of all 3 models side-by-side
- [ ] **Conclusion:** which model is most suitable for deployment and why?

---

## 📂 Repository Structure
```text
DataAnalytics-L2-WineQuality/
├── .gitignore                          # Git ignore rules for Python, Jupyter checkpoints, and virtualenvs
├── README.md                           # Project documentation, inspection findings, and architecture
├── requirements.txt                    # Project dependencies
├── Data/
│   ├── winequality-red.csv             # Red wine physicochemical & sensory dataset (1,599 × 12)
│   ├── winequality-white.csv           # White wine physicochemical & sensory dataset (4,898 × 12)
│   └── winequality.names               # UCI dataset citation & attribute documentation
├── Notebooks/
│   ├── 01_wine_quality_inspection.ipynb # Day 1: Data ingestion, schema inspection & class distribution
│   └── 02_wine_quality_eda.ipynb       # Day 2: Exploratory data analysis, distributions, outliers & correlation
└── reports/
    └── figures/
        ├── day1_quality_distribution.png     # Annotated quality score frequency distribution plot
        ├── day2_univariate_distributions.png # 11-panel histogram & KDE distributions
        ├── day2_feature_boxplots.png         # 11-panel Tukey IQR outlier boxplots
        ├── day2_features_vs_quality.png      # Bivariate boxplots of key chemical drivers across quality
        ├── day2_correlation_heatmap.png      # Lower-triangular Pearson correlation matrix heatmap
        └── day2_top_correlations.png         # Ranked feature correlations with sensory quality
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
