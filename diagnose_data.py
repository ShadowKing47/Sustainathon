import pandas as pd
import os
import chardet

def detect_file_encoding(file_path):
    """Detect the encoding of a file."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

def analyze_csv_structure(file_path):
    """Analyze the structure of a CSV file."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {file_path}")
    print('='*80)
    
    try:
        # Detect file encoding
        encoding = detect_file_encoding(file_path)
        print(f"Detected encoding: {encoding}")
        
        # Try to read the file with detected encoding
        try:
            # First, read without header to see raw data
            df = pd.read_csv(file_path, encoding=encoding, header=None, nrows=5)
            print("\nFirst 5 rows (without header):")
            print(df)
            
            # Now try with header
            df = pd.read_csv(file_path, encoding=encoding)
            print("\nFile read with headers:")
            print(f"Shape: {df.shape}")
            print("\nColumns:")
            for i, col in enumerate(df.columns, 1):
                print(f"  {i}. {col} (dtype: {df[col].dtype})")
            
            print("\nFirst 3 rows of data:")
            print(df.head(3).to_string())
            
            # Check for missing values
            print("\nMissing values per column:")
            print(df.isnull().sum())
            
            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                print("\nNumeric columns summary:")
                print(df[numeric_cols].describe().to_string())
            
            return True
            
        except Exception as e:
            print(f"\nError reading file: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return False

def main():
    # Directory containing the data files
    data_dir = "DataSets"
    
    # List of data files to analyze
    data_files = [
        "energy_consumption_2023.csv",
        "Cost_n_Billing_data.csv",
        "Equipment_n_Appliance_data.csv",
        "IOT_and_sensor_Data.csv",
        "Updated_Environmental & External Factors.csv",
        "Updates_Cost_n_billing.csv"
    ]
    
    # Analyze each file
    for file_name in data_files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            analyze_csv_structure(file_path)
        else:
            print(f"\nFile not found: {file_path}")
    
    print("\nDiagnostic complete!")

if __name__ == "__main__":
    main()
