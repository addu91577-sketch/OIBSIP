"""
Data Cleaning & Quality Audit Utilities for Cafe Sales Pipeline
================================================================
Reusable functions for:
- Data quality audits (nulls, pseudonulls, dtypes, key integrity)
- Pseudonull anomaly cataloging ('ERROR', 'UNKNOWN', whitespace)
- Defective frequency profiling across transactional attributes
- Record-level defect density & co-occurrence matrix computations
- Publication-quality visualization generation for data quality reports
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_quality_report(df: pd.DataFrame, dirty_tokens=None) -> pd.DataFrame:
    """
    Computes a forensic data health report across all columns.
    Identifies true nulls, pseudonull tokens (e.g. 'UNKNOWN', 'ERROR'),
    data types, and defective rates.
    """
    if dirty_tokens is None:
        dirty_tokens = ['UNKNOWN', 'ERROR', 'NULL', 'NONE', 'NAN', 'N/A', '?']

    report_data = []
    total_rows = len(df)

    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        
        # Count string representations of missingness/errors
        dirty_count = 0
        if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            str_series = series.dropna().astype(str).str.strip().str.upper()
            dirty_count = int(str_series.isin(dirty_tokens).sum())
        
        combined_missing = null_count + dirty_count
        missing_pct = (combined_missing / total_rows) * 100 if total_rows > 0 else 0.0

        report_data.append({
            'Column': col,
            'Data Type': str(series.dtype),
            'True Nulls': null_count,
            'Pseudonulls (ERROR/UNKNOWN)': dirty_count,
            'Total Dirty/Missing': combined_missing,
            'Defective Rate (%)': round(missing_pct, 2),
            'Unique Count': series.nunique()
        })

    return pd.DataFrame(report_data)


def catalog_pseudonull_frequencies(df: pd.DataFrame, dirty_tokens=None) -> pd.DataFrame:
    """
    Catalogs granular token counts for specific pseudonull values
    such as 'ERROR' and 'UNKNOWN' across all columns.
    """
    if dirty_tokens is None:
        dirty_tokens = ['ERROR', 'UNKNOWN']

    catalog_data = []
    total_rows = len(df)

    for col in df.columns:
        s = df[col]
        true_nulls = int(s.isnull().sum())
        str_series = s.dropna().astype(str).str.strip().str.upper()
        
        c_error = int((str_series == 'ERROR').sum())
        c_unknown = int((str_series == 'UNKNOWN').sum())
        total_dirty = true_nulls + c_error + c_unknown
        dirty_pct = (total_dirty / total_rows) * 100 if total_rows > 0 else 0.0

        catalog_data.append({
            'Column': col,
            'Raw Dtype': str(s.dtype),
            'True Nulls (NaN)': true_nulls,
            'ERROR Tokens': c_error,
            'UNKNOWN Tokens': c_unknown,
            'Total Dirty Records': total_dirty,
            'Defective Rate (%)': round(dirty_pct, 2)
        })

    catalog_df = pd.DataFrame(catalog_data)
    return catalog_df.sort_values(by='Total Dirty Records', ascending=False).reset_index(drop=True)


def get_defect_masks(df: pd.DataFrame, dirty_tokens=None):
    """
    Generates boolean masks for true nulls, pseudonulls, and combined defects.
    """
    if dirty_tokens is None:
        dirty_tokens = ['ERROR', 'UNKNOWN']

    true_nulls = df.isnull()
    pseudo_nulls = df.apply(
        lambda col: col.dropna().astype(str).str.strip().str.upper().isin(dirty_tokens)
        if (pd.api.types.is_string_dtype(col) or col.dtype == 'object')
        else pd.Series(False, index=df.index)
    ).fillna(False)

    combined_defects = true_nulls | pseudo_nulls
    return true_nulls, pseudo_nulls, combined_defects


def analyze_record_defect_counts(df: pd.DataFrame, dirty_tokens=None) -> pd.DataFrame:
    """
    Computes distribution of defect counts per row across all records.
    Returns DataFrame summarizing rows with 0, 1, 2, ... defects.
    """
    _, _, defect_mask = get_defect_masks(df, dirty_tokens)
    defects_per_row = defect_mask.sum(axis=1)
    dist = defects_per_row.value_counts().sort_index()

    summary_records = []
    total_records = len(df)
    for defect_count, row_count in dist.items():
        summary_records.append({
            'Defects Per Record': defect_count,
            'Record Count': row_count,
            'Percentage (%)': round((row_count / total_records) * 100, 2),
            'Cumulative Records': dist.loc[:defect_count].sum(),
            'Cumulative (%)': round((dist.loc[:defect_count].sum() / total_records) * 100, 2)
        })

    return pd.DataFrame(summary_records)


def compute_defect_cooccurrence(df: pd.DataFrame, dirty_tokens=None) -> pd.DataFrame:
    """
    Calculates pairwise co-occurrence matrix of defects across all columns.
    """
    _, _, defect_mask = get_defect_masks(df, dirty_tokens)
    co_matrix = defect_mask.astype(int).T.dot(defect_mask.astype(int))
    return co_matrix


def plot_defect_distributions(df: pd.DataFrame, output_dir: str = None) -> dict:
    """
    Generates high-resolution visualization suite for data quality reporting:
    1. day3_column_defect_rates.png
    2. day3_record_defect_distribution.png
    3. day3_defect_cooccurrence_matrix.png
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    total_rows = len(df)
    true_nulls, pseudo_nulls, defect_mask = get_defect_masks(df)

    # 1. Figure 1: Column Defect Rates
    col_stats = []
    for col in df.columns:
        tn = int(true_nulls[col].sum())
        pn = int(pseudo_nulls[col].sum())
        tot = int(defect_mask[col].sum())
        col_stats.append({
            'Column': col,
            'True Nulls': tn,
            'Pseudonulls': pn,
            'Total Defects': tot,
            'Defect Rate': (tot / total_rows) * 100
        })
    cdf = pd.DataFrame(col_stats).sort_values(by='Total Defects', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    y_pos = np.arange(len(cdf))
    bar_width = 0.55

    ax.barh(y_pos, cdf['True Nulls'], bar_width, label='True Nulls (NaN)', color='#3498db', alpha=0.9)
    ax.barh(y_pos, cdf['Pseudonulls'], bar_width, left=cdf['True Nulls'], label='Pseudonulls (ERROR/UNKNOWN)', color='#e74c3c', alpha=0.9)

    for idx, (tot, rate) in enumerate(zip(cdf['Total Defects'], cdf['Defect Rate'])):
        ax.text(tot + 60, idx, f'{tot:,} ({rate:.1f}%)', va='center', ha='left', fontsize=9.5, fontweight='bold', color='#2c3e50')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cdf['Column'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Defective Cell Count (Records)', fontsize=11, fontweight='bold')
    ax.set_title('Cafe Sales: Forensic Data Defect Audit by Column (10,000 Records)', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.set_xlim(0, 4700)
    plt.tight_layout()

    saved_paths = {}
    if output_dir:
        p1 = os.path.join(output_dir, 'day3_column_defect_rates.png')
        plt.savefig(p1, dpi=300)
        saved_paths['column_defects'] = p1
    plt.close()

    # 2. Figure 2: Record Defect Distribution & Donut
    row_defects = defect_mask.sum(axis=1)
    dist = row_defects.value_counts().sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300, gridspec_kw={'width_ratios': [1.3, 1]})

    bars = ax1.bar(dist.index, dist.values, color=['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c', '#9b59b6'], edgecolor='black', alpha=0.88, width=0.6)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_rows) * 100
        ax1.text(bar.get_x() + bar.get_width()/2, h + 80, f'{h:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Number of Defective Attributes per Row', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Transaction Count', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Defects per Transaction Record', fontsize=12, fontweight='bold')
    ax1.set_xticks(dist.index)
    ax1.set_ylim(0, 4800)

    clean_cnt = (row_defects == 0).sum()
    dirty_cnt = (row_defects > 0).sum()
    ax2.pie([clean_cnt, dirty_cnt], labels=[f'Pristine (0 Defects)\n{clean_cnt:,} ({clean_cnt/total_rows*100:.1f}%)',
                                            f'Corrupted (≥1 Defect)\n{dirty_cnt:,} ({dirty_cnt/total_rows*100:.1f}%)'],
            colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=140, pctdistance=0.75,
            explode=(0.05, 0), wedgeprops=dict(width=0.45, edgecolor='black'))
    ax2.set_title('Dataset Integrity: Clean vs Defective Rows\n(69.1% Loss Risk under Naive Deletion)', fontsize=12, fontweight='bold')

    plt.suptitle('Cafe Sales: Record-Level Defect Density & Survivorship Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if output_dir:
        p2 = os.path.join(output_dir, 'day3_record_defect_distribution.png')
        plt.savefig(p2, dpi=300)
        saved_paths['record_distribution'] = p2
    plt.close()

    # 3. Figure 3: Defect Co-Occurrence Heatmap
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)
    co_occurrence = defect_mask.astype(int).T.dot(defect_mask.astype(int))
    sns.heatmap(co_occurrence, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Co-Occurring Defective Records'},
                square=True, linewidths=0.8, linecolor='white', ax=ax)
    ax.set_title('Defect Co-Occurrence Matrix across Attributes', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()

    if output_dir:
        p3 = os.path.join(output_dir, 'day3_defect_cooccurrence_matrix.png')
        plt.savefig(p3, dpi=300)
        saved_paths['cooccurrence_matrix'] = p3
    plt.close()

    return saved_paths
