import pandas as pd
import numpy as np
import re

def extract_numeric_features(df):
    """
    Extracts numeric tokens from business_address_clean without removing them.
    """
    if 'business_address_clean' not in df.columns:
        return df
        
    mask = df['business_address_clean'].notna()
    
    # Extract all numbers as a list of strings
    df['address_numbers'] = pd.Series(dtype='object')
    df.loc[mask, 'address_numbers'] = df.loc[mask, 'business_address_clean'].apply(
        lambda x: ' '.join(re.findall(r'\b\d+\b', str(x)))
    )
    
    # Count of numeric tokens
    df['numeric_token_count'] = 0
    df.loc[mask, 'numeric_token_count'] = df.loc[mask, 'address_numbers'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() else 0
    )
    
    # Boolean flag
    df['has_numeric_token'] = (df['numeric_token_count'] > 0).astype(int)
    
    return df

def generate_text_statistics(df):
    """
    Generates length and token count statistics for name and address.
    Also creates token lists.
    """
    # Name features
    if 'business_name_clean' in df.columns:
        mask_name = df['business_name_clean'].notna()
        
        # Token lists (space delimited)
        df['name_tokens'] = pd.Series(dtype='object')
        df.loc[mask_name, 'name_tokens'] = df.loc[mask_name, 'business_name_clean'].apply(
            lambda x: ' '.join(str(x).split())
        )
        
        df['name_length'] = 0
        df.loc[mask_name, 'name_length'] = df.loc[mask_name, 'business_name_clean'].str.len()
        
        df['name_token_count'] = 0
        df.loc[mask_name, 'name_token_count'] = df.loc[mask_name, 'business_name_clean'].apply(
            lambda x: len(str(x).split())
        )
        
    # Address features
    if 'business_address_clean' in df.columns:
        mask_addr = df['business_address_clean'].notna()
        
        df['address_tokens'] = pd.Series(dtype='object')
        df.loc[mask_addr, 'address_tokens'] = df.loc[mask_addr, 'business_address_clean'].apply(
            lambda x: ' '.join(str(x).split())
        )
        
        df['address_length'] = 0
        df.loc[mask_addr, 'address_length'] = df.loc[mask_addr, 'business_address_clean'].str.len()
        
        df['address_token_count'] = 0
        df.loc[mask_addr, 'address_token_count'] = df.loc[mask_addr, 'business_address_clean'].apply(
            lambda x: len(str(x).split())
        )
        
    return df
