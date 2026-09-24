import pandas as pd

# Internal dictionary for address abbreviation expansion
ADDRESS_ABBREVIATIONS = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bhwy\b': 'highway',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bsq\b': 'square',
    r'\bpl\b': 'place',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\bfl\b': 'floor',
    r'\bdept\b': 'department',
    r'\bpkwy\b': 'parkway',
    r'\bmt\b': 'mount',
    r'\bctr\b': 'center',
}

def clean_business_address(df):
    """
    Expands common address abbreviations in business_address_clean.
    """
    if 'business_address_clean' not in df.columns:
        return df
        
    mask = df['business_address_clean'].notna()
    
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        df.loc[mask, 'business_address_clean'] = df.loc[mask, 'business_address_clean'].str.replace(
            pattern, replacement, regex=True
        )
        
    # Normalize spaces again just in case
    df.loc[mask, 'business_address_clean'] = df.loc[mask, 'business_address_clean'].str.replace(
        r'\s+', ' ', regex=True
    ).str.strip()
        
    return df
