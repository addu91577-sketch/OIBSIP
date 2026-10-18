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
├── README.md                           # Project documentation
├── requirements.txt                    # Project dependencies
├── Data/
│   └── dirty_cafe_sales.csv            # Raw dataset (10,000 rows × 8 columns)
└── Notebooks/
    └── 01_cafe_sales_data_cleaning.ipynb # Jupyter Notebook data cleaning workflow
```

---

## 📊 Dataset Overview
The raw dataset simulates realistic point-of-sale (POS) cafe transaction records containing deliberate data entry anomalies, missing values, and pseudonulls.

| Column Name | Description | Example Raw Values |
| :--- | :--- | :--- |
| **Transaction ID** | Unique transaction identifier | `TXN-00001`, `TXN-00002` |
| **Item** | Menu item purchased | `Coffee`, `Tea`, `Sandwich`, `ERROR`, `UNKNOWN` |
| **Quantity** | Units purchased per item | `1`, `2`, `5`, `NaN`, `ERROR` |
| **Price Per Unit** | Unit price in USD | `2.0`, `1.5`, `NaN`, `UNKNOWN` |
| **Total Spent** | Total transaction cost ($) | `4.0`, `15.0`, `NaN`, `ERROR` |
| **Payment Method** | Mode of customer settlement | `Cash`, `Credit Card`, `Digital Wallet` |
| **Location** | Store fulfillment mode | `In-store`, `Takeaway` |
| **Transaction Date** | Transaction timestamp/date | `2023-01-15`, `UNKNOWN` |

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
