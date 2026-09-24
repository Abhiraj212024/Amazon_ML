import pandas as pd
import os
import glob
import logging

logger = logging.getLogger(__name__)

def load_and_validate_dataset(data_dir):
    """
    Loads all source TSV files in a given directory (S1, S2, S3),
    validates the schema, and adds 'source' column.
    
    Args:
        data_dir (str): Path to directory containing TSV files.
    
    Returns:
        dict: A dictionary mapping source names (e.g. 'source1') to their DataFrames.
    """
    expected_columns = {'entity_id', 'business_name', 'business_address', 'country'}
    
    loaded_data = {}
    
    # We look for files ending with .tsv and containing 'source'
    search_pattern = os.path.join(data_dir, '*source*.tsv')
    file_paths = glob.glob(search_pattern)
    
    if not file_paths:
        logger.warning(f"No source TSV files found in {data_dir}")
        return loaded_data
        
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        source_name = filename.replace('train_', '').replace('test_', '').replace('.tsv', '')
        
        logger.info(f"Loading {filename}...")
        
        # Load the data
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        
        # Schema validation
        actual_columns = set(df.columns)
        if not expected_columns.issubset(actual_columns):
            missing = expected_columns - actual_columns
            raise ValueError(f"File {filename} is missing required columns: {missing}")
            
        # Report basic stats
        num_rows, num_cols = df.shape
        logger.info(f"Rows: {num_rows}, Columns: {num_cols}")
        
        # Validate entity_id duplicates
        dup_count = df['entity_id'].duplicated().sum()
        logger.info(f"Duplicate entity_ids: {dup_count}")
        
        # Report missing values (before cleaning)
        missing_vals = df[list(expected_columns)].isna().sum().to_dict()
        logger.info(f"Missing values: {missing_vals}")
        
        # Report unique country values
        unique_countries = df['country'].dropna().unique()
        logger.info(f"Unique country values (raw): {len(unique_countries)}")
        
        # Prefix validation and source extraction
        # e.g., 'S1-...' -> source is 'S1'
        df['source'] = df['entity_id'].str.extract(r'^(S\d+)-')[0]
        missing_source = df['source'].isna().sum()
        if missing_source > 0:
            logger.warning(f"Found {missing_source} records with invalid entity_id prefixes in {filename}")
            
        loaded_data[source_name] = df
        
    return loaded_data
