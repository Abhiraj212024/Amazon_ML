import pandas as pd
import sys
import os

def _project_root():
    """
    Locate the directory that holds `src/`, searching upward from this file.

    The repository keeps scripts beside `src/`, while the submission package
    places them under `src/` so that all source sits there as the challenge
    requires. Searching upward makes the same file work in both layouts.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.isdir(os.path.join(here, "src", "preprocessing")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    raise RuntimeError("could not locate the project root containing src/matching")


sys.path.insert(0, _project_root())
from src.preprocessing.pipeline import preprocess_dataframe

def test_preprocessing():
    # Sample Mock Data
    data = {
        'entity_id': ['S1-001', 'S2-002', 'S3-003'],
        'business_name': ['Amazon, Pvt. Ltd.', 'Google Corp.', '   NaN   '],
        'business_address': ['123 Main St.', '456 Tech Blvd', 'N/A'],
        'country': ['  USA  ', 'india', 'FRANCE']
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    
    processed_df = preprocess_dataframe(df.copy())
    
    print("\nProcessed DataFrame:")
    print(processed_df[['business_name_clean', 'business_name_core', 'business_address_clean', 'address_numbers', 'country_clean']])
    
    # Assertions
    assert processed_df.loc[0, 'business_name_clean'] == 'amazon pvt ltd'
    assert processed_df.loc[0, 'business_name_core'] == 'amazon'
    assert processed_df.loc[1, 'business_name_core'] == 'google'
    assert pd.isna(processed_df.loc[2, 'business_name_clean'])
    
    assert processed_df.loc[0, 'business_address_clean'] == '123 main street'
    assert processed_df.loc[1, 'business_address_clean'] == '456 tech boulevard'
    assert pd.isna(processed_df.loc[2, 'business_address_clean'])
    
    assert processed_df.loc[0, 'address_numbers'] == '123'
    assert processed_df.loc[1, 'address_numbers'] == '456'
    
    assert processed_df.loc[0, 'country_clean'] == 'Usa'
    assert processed_df.loc[1, 'country_clean'] == 'India'
    assert processed_df.loc[2, 'country_clean'] == 'France'
    
    print("\nAll preprocessing tests passed!")

if __name__ == '__main__':
    test_preprocessing()
