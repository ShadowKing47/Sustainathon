import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

class SimpleEnergyPredictor:
    def __init__(self, data_path):
        """Initialize the Simple Energy Predictor."""
        self.data_path = data_path
        self.data = None
        self.model = LinearRegression()
        
    def load_and_prepare_data(self):
        """Load and prepare the data for modeling."""
        print("Loading and preparing data...")
        
        try:
            # Read the data
            self.data = pd.read_csv(self.data_path)
            print(f"Original data shape: {self.data.shape}")
            
            # Display basic info
            print("\nFirst few rows of the data:")
            print(self.data.head())
            
            # Extract features and target
            # For now, let's assume the last column is the target
            X = self.data.iloc[:, :-1]  # All columns except last
            y = self.data.iloc[:, -1]    # Last column as target
            
            print(f"\nFeatures shape: {X.shape}, Target shape: {y.shape}")
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None, None, None
    
    def train_model(self, X_train, y_train):
        """Train the linear regression model."""
        print("\nTraining the model...")
        try:
            self.model.fit(X_train, y_train)
            print("Model training completed.")
            return True
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model on test data."""
        print("\nEvaluating the model...")
        try:
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Root Mean Squared Error: {rmse:.4f}")
            print(f"R² Score: {r2:.4f}")
            
            # Plot actual vs predicted
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted Energy Consumption')
            
            # Create plots directory if it doesn't exist
            os.makedirs('plots', exist_ok=True)
            plt.savefig('plots/actual_vs_predicted.png')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            return False

def main():
    # Path to the data file
    data_file = 'DataSets/energy_consumption_2023.csv'
    
    # Initialize the predictor
    predictor = SimpleEnergyPredictor(data_file)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = predictor.load_and_prepare_data()
    
    if X_train is not None:
        # Train the model
        if predictor.train_model(X_train, y_train):
            # Evaluate the model
            predictor.evaluate_model(X_test, y_test)
    
    print("\nProcess completed!")

if __name__ == "__main__":
    main()
