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


def standardize_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Day 4 Pipeline: Standardizes all text/string fields in the DataFrame.
    - Strips leading and trailing whitespace across all string/object columns.
    - Enforces canonical casing on categorical fields while preserving pseudonull tokens.
    - Ensures primary key formatting consistency.
    """
    df_clean = df.copy()

    # Strip whitespace across all object / string columns
    for col in df_clean.columns:
        if pd.api.types.is_string_dtype(df_clean[col]) or df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Canonical casing maps (preserving uppercase ERROR/UNKNOWN)
    item_map = {
        'coffee': 'Coffee', 'tea': 'Tea', 'sandwich': 'Sandwich', 'salad': 'Salad',
        'cake': 'Cake', 'juice': 'Juice', 'smoothie': 'Smoothie', 'cookie': 'Cookie'
    }
    payment_map = {
        'cash': 'Cash', 'credit card': 'Credit Card', 'digital wallet': 'Digital Wallet'
    }
    location_map = {
        'in-store': 'In-store', 'takeaway': 'Takeaway'
    }

    if 'Item' in df_clean.columns:
        df_clean['Item'] = df_clean['Item'].apply(
            lambda x: item_map.get(x.lower(), x.title()) if isinstance(x, str) and x.upper() not in ['ERROR', 'UNKNOWN'] else x
        )

    if 'Payment Method' in df_clean.columns:
        df_clean['Payment Method'] = df_clean['Payment Method'].apply(
            lambda x: payment_map.get(x.lower(), x.title()) if isinstance(x, str) and x.upper() not in ['ERROR', 'UNKNOWN'] else x
        )

    if 'Location' in df_clean.columns:
        df_clean['Location'] = df_clean['Location'].apply(
            lambda x: location_map.get(x.lower(), x.title()) if isinstance(x, str) and x.upper() not in ['ERROR', 'UNKNOWN'] else x
        )

    if 'Transaction ID' in df_clean.columns:
        df_clean['Transaction ID'] = df_clean['Transaction ID'].apply(
            lambda x: x.upper() if isinstance(x, str) else x
        )

    return df_clean


def convert_pseudonulls_to_nan(df: pd.DataFrame, dirty_tokens=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Day 4 Pipeline: Replaces literal pseudonull sentinel tokens ('ERROR', 'UNKNOWN')
    with standard IEEE np.nan across all attributes.
    Returns:
        (transformed_df, audit_summary_df)
    """
    if dirty_tokens is None:
        dirty_tokens = ['ERROR', 'UNKNOWN']

    upper_tokens = [t.upper() for t in dirty_tokens]
    df_clean = df.copy()
    audit_records = []
    total_records = len(df_clean)

    for col in df_clean.columns:
        series = df_clean[col]
        raw_nans = int(series.isnull().sum())

        if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            is_error = series.apply(lambda x: str(x).strip().upper() == 'ERROR' if pd.notnull(x) else False)
            is_unknown = series.apply(lambda x: str(x).strip().upper() == 'UNKNOWN' if pd.notnull(x) else False)
            is_sentinel = series.apply(lambda x: str(x).strip().upper() in upper_tokens if pd.notnull(x) else False)

            c_error = int(is_error.sum())
            c_unknown = int(is_unknown.sum())
            c_total = int(is_sentinel.sum())

            # Convert to np.nan
            df_clean.loc[is_sentinel, col] = np.nan
        else:
            c_error = 0
            c_unknown = 0
            c_total = 0

        post_nans = int(df_clean[col].isnull().sum())
        null_rate = (post_nans / total_records) * 100 if total_records > 0 else 0.0

        audit_records.append({
            'Column': col,
            'Raw NaNs': raw_nans,
            'Converted ERROR': c_error,
            'Converted UNKNOWN': c_unknown,
            'Total Replaced': c_total,
            'Post-Day 4 NaNs': post_nans,
            'Post Null Rate (%)': round(null_rate, 2)
        })

    audit_df = pd.DataFrame(audit_records)
    return df_clean, audit_df


def verify_pseudonull_elimination(df: pd.DataFrame, dirty_tokens=None) -> dict:
    """
    Day 4 Pipeline: Verifies that no pseudonull tokens remain in any column
    and that entity integrity is 100% intact.
    """
    if dirty_tokens is None:
        dirty_tokens = ['ERROR', 'UNKNOWN']

    upper_tokens = [t.upper() for t in dirty_tokens]
    checks = {}
    total_remaining_tokens = 0

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            found = int(series.apply(lambda x: str(x).strip().upper() in upper_tokens if pd.notnull(x) else False).sum())
        else:
            found = 0
        checks[col] = found
        total_remaining_tokens += found

    key_unique = int(df['Transaction ID'].nunique()) if 'Transaction ID' in df.columns else None
    key_nulls = int(df['Transaction ID'].isnull().sum()) if 'Transaction ID' in df.columns else None

    is_clean = (total_remaining_tokens == 0) and (key_nulls == 0) and (key_unique == len(df))

    return {
        'status': 'PASSED' if is_clean else 'FAILED',
        'total_remaining_sentinels': total_remaining_tokens,
        'column_sentinel_counts': checks,
        'primary_key_unique': key_unique,
        'primary_key_nulls': key_nulls,
        'total_null_cells': int(df.isnull().sum().sum()),
        'all_assertions_passed': is_clean
    }


def plot_day4_standardization_audit(raw_df: pd.DataFrame, clean_df: pd.DataFrame, output_dir: str = None) -> dict:
    """
    Day 4 Pipeline: Generates diagnostic visual artifacts:
    1. day4_pseudonull_normalization_waterfall.png: Stacked decomposition of raw NaNs vs converted tokens
    2. day4_standardized_categorical_profiles.png: Clean categorical profiles after token elimination
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    saved_paths = {}
    total_rows = len(raw_df)

    # 1. Figure 1: Pseudonull Normalization Waterfall / Decomposition
    cols = [c for c in raw_df.columns if c != 'Transaction ID']
    waterfall_data = []
    for c in cols:
        raw_n = int(raw_df[c].isnull().sum())
        s = raw_df[c].dropna().astype(str).str.strip().str.upper()
        err_c = int((s == 'ERROR').sum())
        unk_c = int((s == 'UNKNOWN').sum())
        tot_post = int(clean_df[c].isnull().sum())
        waterfall_data.append({
            'Column': c,
            'Raw NaNs': raw_n,
            'ERROR Tokens': err_c,
            'UNKNOWN Tokens': unk_c,
            'Total Post NaNs': tot_post,
            'Post Rate': (tot_post / total_rows) * 100
        })
    wdf = pd.DataFrame(waterfall_data).sort_values(by='Total Post NaNs', ascending=True)

    fig, ax = plt.subplots(figsize=(10.5, 6), dpi=300)
    y_pos = np.arange(len(wdf))
    bar_h = 0.55

    ax.barh(y_pos, wdf['Raw NaNs'], bar_h, label='Raw Ingested NaNs', color='#3498db', alpha=0.9)
    ax.barh(y_pos, wdf['ERROR Tokens'], bar_h, left=wdf['Raw NaNs'], label='Converted "ERROR" Tokens', color='#e74c3c', alpha=0.9)
    ax.barh(y_pos, wdf['UNKNOWN Tokens'], bar_h, left=wdf['Raw NaNs'] + wdf['ERROR Tokens'], label='Converted "UNKNOWN" Tokens', color='#e67e22', alpha=0.9)

    for idx, (tot, rate) in enumerate(zip(wdf['Total Post NaNs'], wdf['Post Rate'])):
        ax.text(tot + 65, idx, f'{tot:,} ({rate:.1f}%)', va='center', ha='left', fontsize=9.5, fontweight='bold', color='#2c3e50')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(wdf['Column'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Consolidated Missing Cells (Post-Normalization NaNs)', fontsize=11, fontweight='bold')
    ax.set_title('Day 4: Pseudonull Token Normalization to IEEE NaN by Attribute', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, fontsize=9.5)
    ax.set_xlim(0, 4700)
    plt.tight_layout()

    if output_dir:
        p1 = os.path.join(output_dir, 'day4_pseudonull_normalization_waterfall.png')
        plt.savefig(p1, dpi=300)
        saved_paths['waterfall'] = p1
    plt.close()

    # 2. Figure 2: Standardized Categorical Distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # Item distribution (valid records only)
    item_counts = clean_df['Item'].dropna().value_counts()
    axes[0].barh(item_counts.index, item_counts.values, color='#1abc9c', edgecolor='black', alpha=0.88)
    axes[0].set_title(f'Clean Item Frequencies (N={len(clean_df["Item"].dropna()):,})', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Valid Transactions', fontsize=10, fontweight='bold')
    for idx, val in enumerate(item_counts.values):
        pct = (val / len(clean_df['Item'].dropna())) * 100
        axes[0].text(val + 20, idx, f'{val:,} ({pct:.1f}%)', va='center', fontsize=8.5, fontweight='bold')
    axes[0].set_xlim(0, max(item_counts.values) * 1.25)
    axes[0].invert_yaxis()

    # Payment Method distribution
    pm_counts = clean_df['Payment Method'].dropna().value_counts()
    axes[1].bar(pm_counts.index, pm_counts.values, color=['#9b59b6', '#3498db', '#f1c40f'], edgecolor='black', alpha=0.88, width=0.55)
    axes[1].set_title(f'Clean Payment Methods (N={len(clean_df["Payment Method"].dropna()):,})', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Valid Transactions', fontsize=10, fontweight='bold')
    for idx, val in enumerate(pm_counts.values):
        pct = (val / len(clean_df['Payment Method'].dropna())) * 100
        axes[1].text(idx, val + 60, f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1].set_ylim(0, max(pm_counts.values) * 1.2)

    # Location distribution
    loc_counts = clean_df['Location'].dropna().value_counts()
    axes[2].pie(loc_counts.values, labels=[f'{k}\n({v:,})' for k, v in loc_counts.items()],
                colors=['#e67e22', '#2ecc71'], autopct='%1.1f%%', startangle=120, pctdistance=0.75,
                explode=(0.04, 0), wedgeprops=dict(width=0.45, edgecolor='black'))
    axes[2].set_title(f'Clean Location Shares (N={len(clean_df["Location"].dropna()):,})', fontsize=11, fontweight='bold')

    plt.suptitle('Day 4: Post-Normalization Categorical Profiles (Excluding Pseudonulls)', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()

    if output_dir:
        p2 = os.path.join(output_dir, 'day4_standardized_categorical_profiles.png')
        plt.savefig(p2, dpi=300)
        saved_paths['categorical_profiles'] = p2
    plt.close()

    return saved_paths
