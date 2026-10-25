"""
Source package for Cafe Sales Data Cleaning & Transformation Pipeline
"""

from .utils import (
    generate_quality_report,
    catalog_pseudonull_frequencies,
    get_defect_masks,
    analyze_record_defect_counts,
    compute_defect_cooccurrence,
    plot_defect_distributions,
    standardize_text_fields,
    convert_pseudonulls_to_nan,
    verify_pseudonull_elimination,
    plot_day4_standardization_audit,
    enforce_strict_dtypes,
    verify_type_integrity,
    plot_day5_type_casting_audit,
    encode_categorical_features,
    audit_memory_reduction,
    verify_categorical_integrity,
    plot_day6_categorical_and_memory_audit,
    impute_menu_pricing_and_items,
    verify_day7_pricing_integrity,
    plot_day7_pricing_and_item_imputation,
)

__all__ = [
    'generate_quality_report',
    'catalog_pseudonull_frequencies',
    'get_defect_masks',
    'analyze_record_defect_counts',
    'compute_defect_cooccurrence',
    'plot_defect_distributions',
    'standardize_text_fields',
    'convert_pseudonulls_to_nan',
    'verify_pseudonull_elimination',
    'plot_day4_standardization_audit',
    'enforce_strict_dtypes',
    'verify_type_integrity',
    'plot_day5_type_casting_audit',
    'encode_categorical_features',
    'audit_memory_reduction',
    'verify_categorical_integrity',
    'plot_day6_categorical_and_memory_audit',
    'impute_menu_pricing_and_items',
    'verify_day7_pricing_integrity',
    'plot_day7_pricing_and_item_imputation',
]
