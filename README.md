# Business Entity Resolution Preprocessing Pipeline

This directory contains the data preprocessing pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task. 
This pipeline focuses purely on cleaning and normalizing text fields without using any external databases, APIs, or libraries outside of pandas/numpy.

## Preprocessing Steps Included
1. **Data Loading & Validation**: Safely reads Source 1, 2, and 3 TSV files, validates schemas and extracts source origin.
2. **Missing Value Handling**: Converts 'NaN', 'N/A', etc., into standard `pd.NA` for internal consistency, adding `name_missing`, `address_missing`, and `country_missing` flags.
3. **Unicode & Whitespace Normalization**: Uses NFKC normalization and controls repeating whitespace/newlines, maintaining clean string encodings.
4. **Business Name Cleaning**: Generates `business_name_clean` and `business_name_core` (with common legal suffixes removed).
5. **Address Cleaning**: Expands common address abbreviations (e.g. 'rd' -> 'road') into `business_address_clean`.
6. **Country Handling**: Title-cases country strings in `country_clean`.
7. **Feature Extraction**: Generates length features, token counts, and extracts numeric sequences to preserve discriminative information like house/postal numbers.

## Non-Destructive Principles
The original `business_name`, `business_address`, and `country` values are strictly preserved as per challenge requirements. All transformations output to new columns (`_clean`, `_core`, etc.). Ground truth data is untouched.

## Setup and Usage

Install requirements:
```bash
pip install -r requirements.txt
```

Run the pipeline:
```bash
python scripts/run_preprocessing.py \
    --train-dir ../dataset/train \
    --test-dir ../dataset/test \
    --output-dir data/processed \
    --report-file reports/preprocessing_report.json
```

## Intentionally Excluded
Because this is strictly the preprocessing step, the following have been excluded for later stages (Blocking/Candidate Generation/Machine Learning):
- TF-IDF or Embedding generation
- Blocking / Candidate Pair generation
- NER Model Training
- Clustering or Pairwise matching models (XGBoost/LightGBM)
- External data requests or Google Maps Geocoding

The cleaned dataset produced here (in Parquet format) is designed to be directly ingested by the Candidate Generation pipeline.
