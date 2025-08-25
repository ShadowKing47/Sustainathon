"""
Anomaly Detection Module

This module provides functions for detecting unusual energy consumption patterns
using statistical methods and machine learning algorithms.
"""
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """Class for detecting anomalies in energy consumption data."""
    
    def __init__(self, data: pd.DataFrame, timestamp_col: str = 'timestamp', 
                 consumption_col: str = 'consumption', occupancy_col: str = 'occupancy'):
        """
        Initialize the AnomalyDetector.
        
        Args:
            data: DataFrame containing the energy consumption data
            timestamp_col: Name of the timestamp column
            consumption_col: Name of the energy consumption column
            occupancy_col: Name of the occupancy column
        """
        self.data = data.copy()
        self.timestamp_col = timestamp_col
        self.consumption_col = consumption_col
        self.occupancy_col = occupancy_col
        
        # Ensure timestamp is in datetime format
        if self.timestamp_col in self.data.columns:
            self.data[self.timestamp_col] = pd.to_datetime(self.data[self.timestamp_col])
            self.data = self.data.set_index(self.timestamp_col).sort_index()
    
    def detect_statistical_anomalies(self, method: str = 'zscore', 
                                   threshold: float = 3.0, 
                                   window: int = 24) -> pd.DataFrame:
        """
        Detect anomalies using statistical methods.
        
        Args:
            method: Detection method ('zscore', 'iqr', 'rolling_stats')
            threshold: Threshold for anomaly detection
            window: Window size for rolling statistics
            
        Returns:
            DataFrame with anomaly scores and flags
        """
        if self.consumption_col not in self.data.columns:
            raise ValueError(f"Consumption column '{self.consumption_col}' not found in data")
            
        result = self.data[[self.consumption_col]].copy()
        
        if method == 'zscore':
            # Z-score method
            result['zscore'] = stats.zscore(result[self.consumption_col])
            result['anomaly'] = np.abs(result['zscore']) > threshold
            result['anomaly_score'] = np.abs(result['zscore'])
            
        elif method == 'iqr':
            # IQR method
            q1 = result[self.consumption_col].quantile(0.25)
            q3 = result[self.consumption_col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (threshold * iqr)
            upper_bound = q3 + (threshold * iqr)
            
            result['anomaly'] = ~result[self.consumption_col].between(lower_bound, upper_bound)
            # Calculate normalized distance from bounds as score
            result['anomaly_score'] = result.apply(
                lambda x: max(x[self.consumption_col] - upper_bound, 
                             lower_bound - x[self.consumption_col], 0) / iqr 
                if not np.isnan(x[self.consumption_col]) else 0, axis=1
            )
            
        elif method == 'rolling_stats':
            # Rolling statistics method
            rolling_mean = result[self.consumption_col].rolling(window=window).mean()
            rolling_std = result[self.consumption_col].rolling(window=window).std()
            
            upper_bound = rolling_mean + (threshold * rolling_std)
            lower_bound = rolling_mean - (threshold * rolling_std)
            
            result['rolling_mean'] = rolling_mean
            result['upper_bound'] = upper_bound
            result['lower_bound'] = lower_bound
            result['anomaly'] = ~result[self.consumption_col].between(
                result['lower_bound'], result['upper_bound']
            )
            result['anomaly_score'] = result.apply(
                lambda x: max(x[self.consumption_col] - x['upper_bound'], 
                             x['lower_bound'] - x[self.consumption_col], 0) / x['rolling_mean']
                if not np.isnan(x[self.consumption_col]) and x['rolling_mean'] > 0 else 0, axis=1
            )
            
        return result
    
    def detect_machine_learning_anomalies(self, method: str = 'isolation_forest',
                                        contamination: float = 0.05,
                                        n_neighbors: int = 20) -> pd.DataFrame:
        """
        Detect anomalies using machine learning algorithms.
        
        Args:
            method: Detection method ('isolation_forest', 'lof')
            contamination: Expected proportion of outliers in the data
            n_neighbors: Number of neighbors for LOF
            
        Returns:
            DataFrame with anomaly scores and flags
        """
        if self.consumption_col not in self.data.columns:
            raise ValueError(f"Consumption column '{self.consumption_col}' not found in data")
            
        # Prepare features
        features = [self.consumption_col]
        
        # Add time-based features
        if self.timestamp_col in self.data.columns:
            self.data['hour'] = self.data.index.hour
            self.data['dayofweek'] = self.data.index.dayofweek
            self.data['month'] = self.data.index.month
            features.extend(['hour', 'dayofweek', 'month'])
        
        # Add occupancy if available
        if self.occupancy_col in self.data.columns:
            features.append(self.occupancy_col)
        
        # Select and scale features
        X = self.data[features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply anomaly detection
        if method == 'isolation_forest':
            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            predictions = model.fit_predict(X_scaled)
            scores = -model.score_samples(X_scaled)  # Higher is more anomalous
            
        elif method == 'lof':
            model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                novelty=False
            )
            predictions = model.fit_predict(X_scaled)
            scores = -model.negative_outlier_factor_  # Higher is more anomalous
            
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Create result DataFrame
        result = self.data[[self.consumption_col]].copy()
        result['anomaly'] = predictions == -1
        result['anomaly_score'] = scores
        
        return result
    
    def detect_occupancy_anomalies(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Detect anomalies based on occupancy patterns.
        
        Args:
            threshold: Threshold for low consumption during occupancy
            
        Returns:
            DataFrame with occupancy-based anomaly flags
        """
        if self.consumption_col not in self.data.columns or self.occupancy_col not in self.data.columns:
            raise ValueError("Both consumption and occupancy columns are required")
            
        result = self.data[[self.consumption_col, self.occupancy_col]].copy()
        
        # Calculate consumption percentiles by occupancy
        occupancy_groups = result.groupby(self.occupancy_col)[self.consumption_col].quantile(0.1)
        
        # Flag anomalies (low consumption when occupancy is high)
        result['occupancy_threshold'] = result[self.occupancy_col].map(occupancy_groups)
        result['anomaly'] = (result[self.consumption_col] < result['occupancy_threshold'] * threshold) & \
                           (result[self.occupancy_col] > 0)
        result['anomaly_score'] = np.where(
            result['anomaly'],
            (result['occupancy_threshold'] - result[self.consumption_col]) / result['occupancy_threshold'],
            0
        )
        
        return result
    
    def detect_change_points(self, method: str = 'mean_shift', 
                           threshold: float = 3.0,
                           window: int = 24) -> pd.DataFrame:
        """
        Detect change points in the time series.
        
        Args:
            method: Detection method ('mean_shift', 'variance_shift')
            threshold: Threshold for change point detection
            window: Window size for change point detection
            
        Returns:
            DataFrame with change point flags
        """
        if self.consumption_col not in self.data.columns:
            raise ValueError(f"Consumption column '{self.consumption_col}' not found in data")
            
        result = self.data[[self.consumption_col]].copy()
        
        if method == 'mean_shift':
            # Calculate rolling mean and standard deviation
            rolling_mean = result[self.consumption_col].rolling(window=window).mean()
            rolling_std = result[self.consumption_col].rolling(window=window).std()
            
            # Detect shifts in the mean
            mean_shift = np.abs(rolling_mean.diff())
            result['change_point'] = mean_shift > (threshold * rolling_std)
            result['change_score'] = mean_shift / rolling_std
            
        elif method == 'variance_shift':
            # Calculate rolling variance
            rolling_var = result[self.consumption_col].rolling(window=window).var()
            
            # Detect shifts in variance
            var_shift = np.abs(rolling_var.pct_change())
            result['change_point'] = var_shift > threshold
            result['change_score'] = var_shift
            
        return result
    
    def plot_anomalies(self, data: pd.DataFrame, 
                      plot_components: bool = True,
                      figsize: tuple = (15, 8)) -> plt.Figure:
        """
        Plot the time series with anomalies highlighted.
        
        Args:
            data: DataFrame with anomaly detection results
            plot_components: Whether to plot additional components (bounds, trend, etc.)
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        if self.consumption_col not in data.columns or 'anomaly' not in data.columns:
            raise ValueError("Data must contain consumption and anomaly columns")
            
        plt.figure(figsize=figsize)
        
        # Plot consumption
        plt.plot(data.index, data[self.consumption_col], 
                label='Consumption', color='blue', alpha=0.7)
        
        # Plot anomalies
        anomalies = data[data['anomaly']]
        plt.scatter(anomalies.index, anomalies[self.consumption_col], 
                   color='red', label='Anomaly', alpha=0.7, s=50)
        
        # Plot additional components if available
        if plot_components:
            if 'rolling_mean' in data.columns and 'upper_bound' in data.columns:
                plt.plot(data.index, data['rolling_mean'], 
                        label='Rolling Mean', color='green', linestyle='--', alpha=0.7)
                plt.fill_between(
                    data.index, 
                    data['upper_bound'], 
                    data['lower_bound'], 
                    color='gray', 
                    alpha=0.2,
                    label='Normal Range'
                )
        
        plt.title('Energy Consumption with Anomalies Highlighted')
        plt.xlabel('Time')
        plt.ylabel('Consumption')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        return plt.gcf()
    
    def generate_anomaly_report(self, data: pd.DataFrame) -> Dict:
        """
        Generate a summary report of detected anomalies.
        
        Args:
            data: DataFrame with anomaly detection results
            
        Returns:
            Dictionary with anomaly statistics
        """
        if 'anomaly' not in data.columns:
            raise ValueError("Input data must contain 'anomaly' column")
            
        anomalies = data[data['anomaly']]
        total_points = len(data)
        anomaly_count = len(anomalies)
        
        report = {
            'total_points': total_points,
            'anomaly_count': anomaly_count,
            'anomaly_percentage': (anomaly_count / total_points) * 100 if total_points > 0 else 0,
            'anomaly_stats': {}
        }
        
        # Add anomaly statistics if available
        if 'anomaly_score' in anomalies.columns:
            report['anomaly_stats']['min_score'] = float(anomalies['anomaly_score'].min())
            report['anomaly_stats']['max_score'] = float(anomalies['anomaly_score'].max())
            report['anomaly_stats']['mean_score'] = float(anomalies['anomaly_score'].mean())
            report['anomaly_stats']['median_score'] = float(anomalies['anomaly_score'].median())
        
        # Add time-based statistics if timestamp is available
        if hasattr(data.index, 'hour'):
            hourly_anomalies = anomalies.groupby(anomalies.index.hour).size()
            report['hourly_distribution'] = hourly_anomalies.to_dict()
            
            daily_anomalies = anomalies.groupby(anomalies.index.dayofweek).size()
            report['daily_distribution'] = daily_anomalies.to_dict()
        
        return report


def detect_anomalies_in_batch(data: pd.DataFrame, 
                            timestamp_col: str = 'timestamp',
                            consumption_col: str = 'consumption',
                            methods: List[str] = None,
                            **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Run multiple anomaly detection methods in batch.
    
    Args:
        data: Input DataFrame
        timestamp_col: Name of the timestamp column
        consumption_col: Name of the consumption column
        methods: List of methods to run (default: ['zscore', 'iqr', 'isolation_forest'])
        **kwargs: Additional arguments to pass to detection methods
        
    Returns:
        Dictionary with method names as keys and results as values
    """
    if methods is None:
        methods = ['zscore', 'iqr', 'isolation_forest']
    
    detector = AnomalyDetector(data, timestamp_col, consumption_col)
    results = {}
    
    for method in methods:
        try:
            if method in ['zscore', 'iqr', 'rolling_stats']:
                results[method] = detector.detect_statistical_anomalies(method=method, **kwargs)
            elif method in ['isolation_forest', 'lof']:
                results[method] = detector.detect_machine_learning_anomalies(method=method, **kwargs)
            elif method == 'occupancy':
                results[method] = detector.detect_occupancy_anomalies(**kwargs)
            elif method in ['mean_shift', 'variance_shift']:
                results[method] = detector.detect_change_points(method=method, **kwargs)
        except Exception as e:
            print(f"Error running {method}: {str(e)}")
    
    return results
