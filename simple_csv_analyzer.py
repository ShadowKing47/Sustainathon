import os
import csv

def analyze_csv(file_path):
    print(f"\n{'='*80}")
    print(f"Analyzing: {os.path.basename(file_path)}")
    print('='*80)
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    # Read first 5 lines
                    reader = csv.reader(f)
                    headers = next(reader, None)
                    print(f"\nEncoding: {encoding}")
                    print("\nHeaders:")
                    for i, header in enumerate(headers, 1):
                        print(f"  {i}. {header}")
                    
                    print("\nFirst 5 rows:")
                    for i, row in enumerate(reader):
                        if i >= 5:
                            break
                        print(f"  {row}")
                    break
                        
            except UnicodeDecodeError:
                print(f"  Failed with {encoding} encoding")
                continue
                
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
    analyze_csv(file_path)
