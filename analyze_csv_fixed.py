import pandas as pd
import os
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

def analyze_file(file_path):
    print(f"\n{'='*80}")
    print(f"Analyzing: {os.path.basename(file_path)}")
    print('='*80)
    
    try:
        # Detect file encoding
        encoding = detect_encoding(file_path)
        print(f"Detected encoding: {encoding}")
        
        # Try to read the file with detected encoding
        try:
            # First try with standard parameters
            df = pd.read_csv(file_path, encoding=encoding, nrows=10)
            print("\nColumns:")
            for col in df.columns:
                print(f"- {col}")
                
            print("\nFirst 5 rows:")
            print(df.head().to_string())
            
            # Try to get basic stats if possible
            try:
                print("\nBasic statistics:")
                print(df.describe().to_string())
            except:
                print("Could not generate statistics - non-numeric data")
                
        except Exception as e:
            print(f"\nError reading with pandas: {str(e)}")
            
            # If standard read fails, try reading raw lines
            print("\nFirst 5 lines of raw content:")
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                for i, line in enumerate(f):
                    if i < 5:
                        print(f"{i+1}: {line.strip()}")
                    else:
                        break
                        
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")

# Directory containing the CSV files
data_dir = r"c:\Users\91995\OneDrive\Desktop\Hackathon\Sustainathon\DataSets"

# List of CSV files to analyze
csv_files = [
    "Cost_n_Billing_data.csv",
    "Equipment_n_Appliance_data.csv",
    "IOT_and_sensor_Data.csv",
    "Updated_Environmental & External Factors.csv",
    "Updates_Cost_n_billing.csv",
    "energy_consumption_2023.csv"
]

# Analyze each file
for csv_file in csv_files:
    file_path = os.path.join(data_dir, csv_file)
    analyze_file(file_path)
