import pandas as pd
import numpy as np
import unicodedata

def handle_missing_values(df, columns):
    """
    Standardize various missing value representations to pd.NA/None.
    """
    missing_representations = [
        'nan', 'null', 'n/a', 'na', 'empty', 'none'
    ]
    
    for col in columns:
        if col in df.columns:
            clean_col = f"{col}_clean"
            if clean_col not in df.columns:
                df[clean_col] = df[col]
            
            # Replace whitespace-only strings with NaN
            df[clean_col] = df[clean_col].replace(r'^\s*$', np.nan, regex=True)
            
            # String matching for explicit missing indicators
            is_missing_str = df[clean_col].astype(str).str.strip().str.lower().isin(missing_representations)
            df.loc[is_missing_str, clean_col] = np.nan
            
            # Create missing flag
            if col == 'business_name':
                df['name_missing'] = df[clean_col].isna().astype(int)
            elif col == 'business_address':
                df['address_missing'] = df[clean_col].isna().astype(int)
            elif col == 'country':
                df['country_missing'] = df[clean_col].isna().astype(int)
                
    return df

def normalize_unicode_and_whitespace(df, columns):
    """
    Apply Unicode normalization (NFKC), lowercasing, and whitespace normalization.
    """
    for col in columns:
        clean_col = f"{col}_clean"
        if clean_col not in df.columns:
            continue
            
        # Only process non-null values
        mask = df[clean_col].notna()
        
        # 1. Unicode normalization (NFKC) & 2. Lowercasing
        df.loc[mask, clean_col] = df.loc[mask, clean_col].apply(
            lambda x: unicodedata.normalize('NFKC', str(x)).lower()
        )
        
        # 3. Punctuation spacing (replace with space where appropriate, e.g., comma, dot)
        # "ABC, Pvt. Ltd." -> "abc pvt ltd"
        punctuation_to_space = r'[,\.\-\(\)\[\]\{\}\"\']'
        df.loc[mask, clean_col] = df.loc[mask, clean_col].str.replace(punctuation_to_space, ' ', regex=True)
        
        # Replace '&' with 'and'
        df.loc[mask, clean_col] = df.loc[mask, clean_col].str.replace(r'\s*&\s*', ' and ', regex=True)
        
        # 4. Whitespace normalization (tabs/newlines to space, multiple spaces to single)
        df.loc[mask, clean_col] = df.loc[mask, clean_col].str.replace(r'\s+', ' ', regex=True)
        
        # 5. Leading/trailing whitespace removal
        df.loc[mask, clean_col] = df.loc[mask, clean_col].str.strip()
        
    return df

def normalize_country(df):
    """
    Normalize country field casing and whitespace.
    """
    if 'country' in df.columns:
        if 'country_clean' not in df.columns:
            df['country_clean'] = df['country']
            
        mask = df['country_clean'].notna()
        df.loc[mask, 'country_clean'] = df.loc[mask, 'country_clean'].astype(str).str.strip().str.title()
        
    return df
