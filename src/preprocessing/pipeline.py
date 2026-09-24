import pandas as pd
from .loader import load_and_validate_dataset
from .normalization import handle_missing_values, normalize_unicode_and_whitespace, normalize_country
from .name_cleaning import create_business_name_core
from .address_cleaning import clean_business_address
from .features import extract_numeric_features, generate_text_statistics
import logging

logger = logging.getLogger(__name__)

def preprocess_dataframe(df):
    """
    Applies the full preprocessing pipeline to a single dataframe.
    """
    # 1. Initialize clean columns to original values
    df['business_name_clean'] = df['business_name']
    df['business_address_clean'] = df['business_address']
    df['country_clean'] = df['country']
    
    # 2. Handle missing values & create flags
    df = handle_missing_values(df, ['business_name', 'business_address', 'country'])
    
    # 3 & 4. Unicode & Basic Text Normalization
    df = normalize_unicode_and_whitespace(df, ['business_name', 'business_address'])
    
    # 5. Business Name Normalization (core generation)
    df = create_business_name_core(df)
    
    # 6. Address Normalization
    df = clean_business_address(df)
    
    # 7 & 10. Country Handling
    df = normalize_country(df)
    
    # 8 & 9. Features Extraction
    df = extract_numeric_features(df)
    df = generate_text_statistics(df)
    
    return df

def generate_quality_report(original_df, processed_df, dataset_name):
    """
    Generates a dictionary with data quality metrics for the report.
    """
    report = {
        'dataset_name': dataset_name,
        'num_records': len(processed_df),
        'unique_entity_ids': processed_df['entity_id'].nunique(),
        'duplicate_entity_ids': int(processed_df['entity_id'].duplicated().sum()),
        'missing_name_count': int(processed_df['name_missing'].sum()),
        'missing_address_count': int(processed_df['address_missing'].sum()),
        'missing_country_count': int(processed_df['country_missing'].sum()),
        'unique_countries': int(processed_df['country_clean'].nunique()) if 'country_clean' in processed_df.columns else 0,
        'avg_name_length': float(processed_df['name_length'].mean()) if 'name_length' in processed_df.columns else 0.0,
        'avg_address_length': float(processed_df['address_length'].mean()) if 'address_length' in processed_df.columns else 0.0,
        'min_name_length': int(processed_df['name_length'].min()) if 'name_length' in processed_df.columns else 0,
        'max_name_length': int(processed_df['name_length'].max()) if 'name_length' in processed_df.columns else 0,
        'min_address_length': int(processed_df['address_length'].min()) if 'address_length' in processed_df.columns else 0,
        'max_address_length': int(processed_df['address_length'].max()) if 'address_length' in processed_df.columns else 0,
    }
    
    # Changed counts (compare string representations to handle NaNs safely)
    name_changed = (original_df['business_name'].astype(str) != processed_df['business_name_clean'].astype(str)).sum()
    address_changed = (original_df['business_address'].astype(str) != processed_df['business_address_clean'].astype(str)).sum()
    country_changed = (original_df['country'].astype(str) != processed_df['country_clean'].astype(str)).sum()
    
    report['name_changed_count'] = int(name_changed)
    report['address_changed_count'] = int(address_changed)
    report['country_changed_count'] = int(country_changed)
    
    return report
