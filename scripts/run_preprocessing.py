import argparse
import os
import json
import logging
import sys

# Ensure src can be imported
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

from src.preprocessing.loader import load_and_validate_dataset
from src.preprocessing.pipeline import preprocess_dataframe, generate_quality_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run Preprocessing Pipeline for Amazon ML Challenge 2026')
    parser.add_argument('--train-dir', type=str, required=True, help='Directory containing raw training data')
    parser.add_argument('--test-dir', type=str, required=True, help='Directory containing raw test data')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save processed data')
    parser.add_argument('--report-file', type=str, default='reports/preprocessing_report.json', help='Path to save quality report')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report_file), exist_ok=True)
    
    reports = []
    
    # Process Train Data
    logger.info("Loading training data...")
    train_data = load_and_validate_dataset(args.train_dir)
    for source_name, df in train_data.items():
        logger.info(f"Preprocessing train {source_name}...")
        processed_df = preprocess_dataframe(df.copy())
        
        report = generate_quality_report(df, processed_df, f"train_{source_name}")
        reports.append(report)
        
        # Output parquet
        out_path = os.path.join(args.output_dir, f"train_{source_name}_processed.parquet")
        processed_df.to_parquet(out_path, index=False)
        logger.info(f"Saved {out_path}")
        
    # Process Test Data
    logger.info("Loading test data...")
    test_data = load_and_validate_dataset(args.test_dir)
    for source_name, df in test_data.items():
        logger.info(f"Preprocessing test {source_name}...")
        processed_df = preprocess_dataframe(df.copy())
        
        report = generate_quality_report(df, processed_df, f"test_{source_name}")
        reports.append(report)
        
        # Output parquet
        out_path = os.path.join(args.output_dir, f"test_{source_name}_processed.parquet")
        processed_df.to_parquet(out_path, index=False)
        logger.info(f"Saved {out_path}")
        
    # Save Report
    with open(args.report_file, 'w') as f:
        json.dump(reports, f, indent=4)
        
    logger.info(f"Preprocessing complete! Report saved to {args.report_file}")
    
if __name__ == '__main__':
    main()
