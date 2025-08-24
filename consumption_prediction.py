import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

class EnergyConsumptionPredictor:
    def __init__(self, data_path):
        """
        Initialize the Energy Consumption Predictor.
        
        Args:
            data_path (str): Path to the energy consumption data file
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and preprocess the energy consumption data."""
        print("Loading data...")
        try:
            # First, read without parsing dates to inspect the data
            sample_df = pd.read_csv(self.data_path, nrows=5)
            print("Sample data before processing:")
            print(sample_df.head())
            
            # Try to identify the timestamp column
            timestamp_col = None
            for col in sample_df.columns:
                if any(term in col.lower() for term in ['time', 'date', 'timestamp']):
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                print("Warning: Could not identify timestamp column. Using first column as timestamp.")
                timestamp_col = sample_df.columns[0]
            
            print(f"\nUsing column '{timestamp_col}' as timestamp")
            
            # Now read the full data with proper date parsing
            try:
                self.data = pd.read_csv(
                    self.data_path,
                    parse_dates=[timestamp_col],
                    index_col=timestamp_col,
                    dayfirst=True,  # Try dayfirst for dates like 1/8/2023
                    infer_datetime_format=True
                )
            except Exception as e:
                print(f"Error with dayfirst=True, trying without: {e}")
                self.data = pd.read_csv(
                    self.data_path,
                    parse_dates=[timestamp_col],
                    index_col=timestamp_col
                )
            
            # Ensure the index is datetime and sort it
            self.data.index = pd.to_datetime(self.data.index)
            self.data = self.data.sort_index()
            
            print(f"\nData loaded successfully. Shape: {self.data.shape}")
            print("\nColumns:", self.data.columns.tolist())
            print("\nData types:")
            print(self.data.dtypes)
            print("\nSample data:")
            print(self.data.head())
            
            # Convert all columns to numeric, coercing errors to NaN
            for col in self.data.select_dtypes(include=['object']).columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Handle missing values
            self.data = self.data.ffill().bfill()  # Forward fill then backward fill
            
            # Drop any remaining NaN values
            self.data = self.data.dropna()
            
            print("\nAfter cleaning:")
            print(f"Data shape: {self.data.shape}")
            print("\nSample data after cleaning:")
            print(self.data.head())
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def create_features(self, target_col='Total_Consumption_kWh', lag_periods=[1, 2, 3, 24, 168]):
        """
        Create time-series features from the data.
        
        Args:
            target_col (str): Name of the target column
            lag_periods (list): List of periods for lag features
        """
        print("\nCreating features...")
        
        # Make a copy of the data
        df = self.data.copy()
        
        # Create time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['is_weekend'] = df.index.dayofweek >= 5
        
        # Create lag features
        for lag in lag_periods:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Create rolling statistics
        df['rolling_mean_24h'] = df[target_col].rolling(window=24).mean().shift(1)
        df['rolling_std_24h'] = df[target_col].rolling(window=24).std().shift(1)
        
        # Drop rows with NaN values created by lag/rolling features
        df = df.dropna()
        
        # Separate features and target
        self.X = df.drop(columns=[target_col])
        self.y = df[target_col]
        
        print(f"Feature engineering complete. X shape: {self.X.shape}, y shape: {self.y.shape}")
    
    def split_data(self, test_size=0.2):
        """
        Split data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data to use for testing
        """
        # For time series, we don't shuffle the data
        train_size = int(len(self.X) * (1 - test_size))
        
        self.X_train, self.X_test = self.X[:train_size], self.X[train_size:]
        self.y_train, self.y_test = self.y[:train_size], self.y[train_size:]
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nData split complete:")
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Testing set: {len(self.X_test)} samples")
    
    def train_baseline_models(self):
        """Train baseline regression models."""
        print("\nTraining baseline models...")
        
        # Initialize models
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
            'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            
            # Store results
            self.results[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
            
            print(f"{name} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}, "
                  f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    def evaluate_models(self):
        """Evaluate and compare all trained models."""
        if not self.results:
            print("No models have been trained yet.")
            return
        
        print("\nModel Evaluation Results:")
        print("-" * 80)
        print(f"{'Model':<20} {'Train RMSE':<15} {'Test RMSE':<15} {'Train R²':<15} {'Test R²':<15}")
        print("-" * 80)
        
        for name, metrics in self.results.items():
            print(f"{name:<20} {metrics['train_rmse']:<15.4f} {metrics['test_rmse']:<15.4f} "
                  f"{metrics['train_r2']:<15.4f} {metrics['test_r2']:<15.4f}")
    
    def plot_predictions(self, model_name, num_points=100):
        """
        Plot actual vs predicted values for a specific model.
        
        Args:
            model_name (str): Name of the model to plot
            num_points (int): Number of points to plot (for clarity)
        """
        if model_name not in self.models:
            print(f"Model '{model_name}' not found.")
            return
        
        # Get predictions for the test set
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        # Plot a subset of points for clarity
        plt.figure(figsize=(15, 6))
        plt.plot(self.y_test.values[:num_points], label='Actual', color='blue')
        plt.plot(y_pred[:num_points], label='Predicted', color='red', alpha=0.7)
        plt.title(f'{model_name} - Actual vs Predicted Energy Consumption')
        plt.xlabel('Time Step')
        plt.ylabel('Energy Consumption (kWh)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{model_name.lower().replace(" ", "_")}_predictions.png')
        plt.close()
    
    def save_models(self, directory='models'):
        """Save trained models to disk."""
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            filename = os.path.join(directory, f'{name.lower().replace(" ", "_")}.joblib')
            joblib.dump(model, filename)
            print(f"Saved {name} to {filename}")
        
        # Also save the scaler
        joblib.dump(self.scaler, os.path.join(directory, 'scaler.joblib'))
        print("Saved scaler")

def main():
    # Initialize the predictor
    data_file = 'DataSets/energy_consumption_2023.csv'  # Update this path as needed
    predictor = EnergyConsumptionPredictor(data_file)
    
    # Load and preprocess data
    if not predictor.load_data():
        print("Failed to load data. Exiting...")
        return
    
    # Create features
    predictor.create_features()
    
    # Split data
    predictor.split_data(test_size=0.2)
    
    # Train models
    predictor.train_baseline_models()
    
    # Evaluate models
    predictor.evaluate_models()
    
    # Generate prediction plots for each model
    for model_name in predictor.models.keys():
        predictor.plot_predictions(model_name)
    
    # Save models
    predictor.save_models()
    
    print("\nModel training and evaluation complete!")

if __name__ == "__main__":
    main()
