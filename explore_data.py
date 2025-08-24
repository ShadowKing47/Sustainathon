import pandas as pd
import os
from pathlib import Path

def explore_file(file_path):
    """Explore a single CSV file and print its structure."""
    print(f"\n{'='*80}")
    print(f"Exploring: {file_path}")
    print('='*80)
    
    try:
        # First, try to read the file with different encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'utf-16']
        df = None
        
        for encoding in encodings:
            try:
                # Try reading without headers first to see the raw data
                df = pd.read_csv(file_path, nrows=5, encoding=encoding, header=None)
                print(f"\nSuccessfully read with encoding: {encoding}")
                print("First 5 rows (raw):")
                print(df)
                
                # Now try with headers
                df = pd.read_csv(file_path, encoding=encoding)
                print("\nFile read with headers:")
                print(f"Shape: {df.shape}")
                print("\nColumns:")
                for i, col in enumerate(df.columns, 1):
                    print(f"  {i}. {col} (dtype: {df[col].dtype})")
                
                print("\nSample data (first 3 rows):")
                print(df.head(3).to_string())
                
                # Try to identify potential timestamp columns
                print("\nPotential timestamp columns:")
                for col in df.columns:
                    if any(term in col.lower() for term in ['time', 'date', 'timestamp']):
                        print(f"  - {col}: {df[col].head(3).tolist()}")
                
                # Basic statistics for numeric columns
                print("\nNumeric columns summary:")
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    print(df[numeric_cols].describe().to_string())
                else:
                    print("No numeric columns found.")
                
                # Check for missing values
                print("\nMissing values per column:")
                print(df.isnull().sum())
                
                return True
                
            except Exception as e:
                print(f"  Failed with {encoding}: {str(e)}")
                continue
        
        print("\nFailed to read file with any encoding.")
        return False
        
    except Exception as e:
        print(f"Error exploring file: {str(e)}")
        return False

def main():
    # Directory containing the CSV files
    data_dir = Path("DataSets")
    
    # List of CSV files to analyze
    csv_files = [
        "Cost_n_Billing_data.csv",
        "Equipment_n_Appliance_data.csv",
        "IOT_and_sensor_Data.csv",
        "Updated_Environmental & External Factors.csv",
        "Updates_Cost_n_billing.csv",
        "energy_consumption_2023.csv"
    ]
    
    # Explore each file
    for csv_file in csv_files:
        file_path = data_dir / csv_file
        if file_path.exists():
            explore_file(str(file_path))
        else:
            print(f"\nFile not found: {file_path}")
    
    print("\nData exploration complete!")

if __name__ == "__main__":
    main()
