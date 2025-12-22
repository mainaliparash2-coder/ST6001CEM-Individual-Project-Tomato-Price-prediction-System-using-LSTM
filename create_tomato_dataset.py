"""
Script to filter the freshForecast dataset to contain only tomato records.
This creates a tomato-focused dataset for price prediction in Kathmandu markets.
"""

import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime

def create_tomato_dataset():
    """Filter dataset to tomato-only records and backup original."""
    
    # Paths
    base_dir = Path(__file__).resolve().parent
    original_dataset = base_dir / 'dataset.csv'
    backup_dataset = base_dir / 'dataset_all_commodities.csv'
    flask_original = base_dir / 'flask_app' / 'dataset.csv'
    flask_backup = base_dir / 'flask_app' / 'dataset_all_commodities.csv'
    
    print("=" * 60)
    print("Tomato Dataset Creation Script")
    print("=" * 60)
    
    # Load original dataset
    print(f"\n1. Loading original dataset from: {original_dataset}")
    df = pd.read_csv(original_dataset)
    print(f"   ✓ Total records: {len(df):,}")
    print(f"   ✓ Total commodities: {df['Commodity'].nunique()}")
    
    # Filter to tomato records only
    print("\n2. Filtering to tomato records...")
    tomato_df = df[df['Commodity'].str.contains('Tomato', case=False, na=False)].copy()
    print(f"   ✓ Tomato records: {len(tomato_df):,}")
    print(f"   ✓ Tomato varieties: {tomato_df['Commodity'].nunique()}")
    
    # Display tomato varieties
    print("\n3. Tomato varieties found:")
    for variety in sorted(tomato_df['Commodity'].unique()):
        count = len(tomato_df[tomato_df['Commodity'] == variety])
        print(f"   - {variety}: {count:,} records")
    
    # Date range
    print("\n4. Date range:")
    tomato_df['Date'] = pd.to_datetime(tomato_df['Date'])
    print(f"   ✓ From: {tomato_df['Date'].min().strftime('%Y-%m-%d')}")
    print(f"   ✓ To: {tomato_df['Date'].max().strftime('%Y-%m-%d')}")
    
    # Price statistics
    print("\n5. Price statistics (NPR per Kg):")
    print(f"   ✓ Minimum price range: {tomato_df['Minimum'].min():.2f} - {tomato_df['Minimum'].max():.2f}")
    print(f"   ✓ Maximum price range: {tomato_df['Maximum'].min():.2f} - {tomato_df['Maximum'].max():.2f}")
    print(f"   ✓ Average price range: {tomato_df['Average'].min():.2f} - {tomato_df['Average'].max():.2f}")
    
    # Backup original datasets
    print("\n6. Creating backups...")
    if not backup_dataset.exists():
        shutil.copy2(original_dataset, backup_dataset)
        print(f"   ✓ Backed up: {backup_dataset}")
    else:
        print(f"   ⚠ Backup already exists: {backup_dataset}")
    
    if not flask_backup.exists():
        shutil.copy2(flask_original, flask_backup)
        print(f"   ✓ Backed up: {flask_backup}")
    else:
        print(f"   ⚠ Backup already exists: {flask_backup}")
    
    # Save tomato-only dataset
    print("\n7. Saving tomato-only dataset...")
    tomato_df.to_csv(original_dataset, index=False)
    print(f"   ✓ Saved: {original_dataset}")
    
    tomato_df.to_csv(flask_original, index=False)
    print(f"   ✓ Saved: {flask_original}")
    
    # Validation
    print("\n8. Validating filtered dataset...")
    validation_df = pd.read_csv(original_dataset)
    all_tomato = validation_df['Commodity'].str.contains('Tomato', case=False, na=False).all()
    
    if all_tomato and len(validation_df) == len(tomato_df):
        print("   ✓ Validation passed: Dataset contains only tomato records")
    else:
        print("   ✗ Validation failed!")
        return False
    
    print("\n" + "=" * 60)
    print("SUCCESS: Tomato dataset created successfully!")
    print("=" * 60)
    print(f"\nOriginal dataset backed up to:")
    print(f"  - {backup_dataset}")
    print(f"  - {flask_backup}")
    print(f"\nNew tomato-only dataset saved to:")
    print(f"  - {original_dataset}")
    print(f"  - {flask_original}")
    print(f"\nTotal tomato records: {len(tomato_df):,}")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = create_tomato_dataset()
    exit(0 if success else 1)
