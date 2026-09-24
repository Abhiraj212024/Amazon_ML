import pandas as pd
import re

# Internal dictionary of common legal suffixes
LEGAL_SUFFIXES = {
    r'\bcorp\b': '',
    r'\bcorporation\b': '',
    r'\binc\b': '',
    r'\bincorporated\b': '',
    r'\bllc\b': '',
    r'\bltd\b': '',
    r'\blimited\b': '',
    r'\bpvt\b': '',
    r'\bprivate\b': '',
    r'\bco\b': '',
    r'\bcompany\b': '',
    r'\bplc\b': '',
}

def create_business_name_core(df):
    """
    Creates business_name_core from business_name_clean by removing common legal suffixes.
    """
    if 'business_name_clean' not in df.columns:
        return df
        
    df['business_name_core'] = df['business_name_clean']
    mask = df['business_name_core'].notna()
    
    # Apply suffix replacements
    for pattern, replacement in LEGAL_SUFFIXES.items():
        df.loc[mask, 'business_name_core'] = df.loc[mask, 'business_name_core'].str.replace(
            pattern, replacement, regex=True
        )
        
    # Clean up any resulting double spaces or trailing spaces
    df.loc[mask, 'business_name_core'] = df.loc[mask, 'business_name_core'].str.replace(
        r'\s+', ' ', regex=True
    ).str.strip()
    
    return df
