import pandas as pd
import os

# Directory containing the CSV files
data_dir = r"c:\Users\91995\OneDrive\Desktop\Hackathon\Sustainathon\DataSets"

# Function to analyze a CSV file
def analyze_csv(file_name):
    file_path = os.path.join(data_dir, file_name)
    print(f"\n{'='*80}\nAnalyzing: {file_name}\n{'='*80}")
    
    try:
        # Read the first few rows to get column info
        df = pd.read_csv(file_path, nrows=5)
        
        # Display basic information
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        for col in df.columns:
            print(f"- {col}")
            
        print("\nFirst few rows:")
        print(df.head())
        
        # Try to read more data for better analysis
        try:
            df_full = pd.read_csv(file_path, nrows=20)
            print("\nSample values from first 20 rows:")
            for col in df_full.columns:
                print(f"\n{col}:")
                print(df_full[col].value_counts().head())
        except Exception as e:
            print(f"\nCould not read full sample: {str(e)}")
            
    except Exception as e:
        print(f"Error reading file: {str(e)}")

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
    analyze_csv(csv_file)
