"""
Data Cleaning & Quality Audit Utilities for Cafe Sales Pipeline
================================================================
Reusable functions for:
- Data quality audits (nulls, pseudonulls, dtypes, key integrity)
- Pseudonull anomaly cataloging ('ERROR', 'UNKNOWN', whitespace)
- Defective frequency profiling across transactional attributes
"""

import os
import pandas as pd
import numpy as np


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
