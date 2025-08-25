"""
Anomaly detection module for the Energy Monitoring System.
"""
from typing import Dict, List, Optional, Union, Tuple, Callable
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging
from datetime import datetime, timedelta
import warnings

# Import config
from .config_loader import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class AnomalyDetector:
    """
    A class for detecting anomalies in energy consumption data.
    Supports multiple detection algorithms and provides evaluation metrics.
    """
    
    def __init__(self, method: str = 'isolation_forest', **kwargs):
        """
        Initialize the AnomalyDetector with the specified method.
        
        Args:
            method: Anomaly detection method. One of:
                   - 'isolation_forest': Isolation Forest algorithm
                   - 'one_class_svm': One-Class SVM algorithm
                   - 'elliptic_envelope': Elliptic Envelope algorithm
                   - 'lof': Local Outlier Factor algorithm
                   - 'zscore': Z-Score based detection
                   - 'iqr': Interquartile Range based detection
            **kwargs: Additional arguments for the specific detection method
        """
        self.method = method.lower()
        self.model = None
        self.scaler = None
        self.fitted = False
        self.params = kwargs
        
        # Initialize the appropriate model
        if self.method == 'isolation_forest':
            self.model = IsolationForest(
                n_estimators=kwargs.get('n_estimators', 100),
                max_samples=kwargs.get('max_samples', 'auto'),
                contamination=kwargs.get('contamination', 0.01),
                max_features=kwargs.get('max_features', 1.0),
                bootstrap=kwargs.get('bootstrap', False),
                n_jobs=kwargs.get('n_jobs', -1),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.method == 'one_class_svm':
            self.model = OneClassSVM(
                kernel=kwargs.get('kernel', 'rbf'),
                gamma=kwargs.get('gamma', 'scale'),
                nu=kwargs.get('nu', 0.01),
                degree=kwargs.get('degree', 3),
                cache_size=kwargs.get('cache_size', 200)
            )
        elif self.method == 'elliptic_envelope':
            self.model = EllipticEnvelope(
                store_precision=kwargs.get('store_precision', True),
                assume_centered=kwargs.get('assume_centered', False),
                support_fraction=kwargs.get('support_fraction', None),
                contamination=kwargs.get('contamination', 0.01),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.method == 'lof':
            self.model = LocalOutlierFactor(
                n_neighbors=kwargs.get('n_neighbors', 20),
                algorithm=kwargs.get('algorithm', 'auto'),
                leaf_size=kwargs.get('leaf_size', 30),
                metric=kwargs.get('metric', 'minkowski'),
                p=kwargs.get('p', 2),
                metric_params=kwargs.get('metric_params', None),
                contamination=kwargs.get('contamination', 0.01),
                n_jobs=kwargs.get('n_jobs', -1)
            )
        elif self.method in ['zscore', 'iqr']:
            # These are rule-based methods, no model to initialize
            self.model = None
        else:
            raise ValueError(f"Unsupported anomaly detection method: {method}")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None) -> 'AnomalyDetector':
        """
        Fit the anomaly detection model to the data.
        
        Args:
            X: Input features (DataFrame or numpy array)
            y: Optional target variable (ignored for unsupervised methods)
            
        Returns:
            self: Returns the instance itself
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
        
        # Scale the data if needed
        if self.method in ['isolation_forest', 'one_class_svm', 'elliptic_envelope', 'lof']:
            self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Fit the model if it's a scikit-learn estimator
        if hasattr(self.model, 'fit'):
            if self.method == 'lof':
                # LOF doesn't have a separate fit method
                self.fitted = True
            else:
                self.model.fit(X_scaled)
                self.fitted = True
        else:
            # For rule-based methods, just mark as fitted
            self.fitted = True
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict anomalies in the data.
        
        Args:
            X: Input features (DataFrame or numpy array)
            
        Returns:
            Array of anomaly scores (higher values indicate more anomalous)
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        # Scale the data if a scaler was fitted
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_values)
        else:
            X_scaled = X_values
        
        # Get anomaly scores based on the method
        if self.method == 'isolation_forest':
            # Lower scores are more anomalous
            scores = -self.model.decision_function(X_scaled)
        elif self.method == 'one_class_svm':
            # Negative scores are more anomalous
            scores = -self.model.decision_function(X_scaled)
        elif self.method == 'elliptic_envelope':
            # Negative scores are more anomalous
            scores = -self.model.decision_function(X_scaled)
        elif self.method == 'lof':
            # Negative scores are more anomalous
            scores = -self.model.fit_predict(X_scaled)
        elif self.method == 'zscore':
            # Calculate z-scores for each feature and take the maximum absolute value
            means = np.mean(X_scaled, axis=0)
            stds = np.std(X_scaled, axis=0, ddof=1)
            stds[stds == 0] = 1  # Avoid division by zero
            z_scores = np.abs((X_scaled - means) / stds)
            scores = np.max(z_scores, axis=1)
        elif self.method == 'iqr':
            # Calculate IQR-based scores
            q1 = np.percentile(X_scaled, 25, axis=0)
            q3 = np.percentile(X_scaled, 75, axis=0)
            iqr = q3 - q1
            iqr[iqr == 0] = 1  # Avoid division by zero
            
            # Calculate modified Z-scores
            medians = np.median(X_scaled, axis=0)
            mad = 1.4826 * np.median(np.abs(X_scaled - medians), axis=0)
            mad[mad == 0] = 1  # Avoid division by zero
            modified_z_scores = 0.6745 * (X_scaled - medians) / mad
            scores = np.max(np.abs(modified_z_scores), axis=1)
        
        return scores
    
    def detect_anomalies(self, X: Union[pd.DataFrame, np.ndarray], 
                         threshold: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Detect anomalies in the data based on the fitted model.
        
        Args:
            X: Input features (DataFrame or numpy array)
            threshold: Threshold for anomaly detection. If None, uses a default
                     based on the method (e.g., 3.0 for zscore, 1.5 for IQR)
            
        Returns:
            Dictionary containing:
                - 'scores': Anomaly scores for each sample
                - 'is_anomaly': Boolean array indicating anomalies
                - 'threshold': Threshold used for detection
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Get anomaly scores
        scores = self.predict(X)
        
        # Determine threshold if not provided
        if threshold is None:
            if self.method == 'zscore':
                threshold = 3.0
            elif self.method == 'iqr':
                threshold = 1.5
            else:
                # For model-based methods, use the contamination parameter if available
                if hasattr(self.model, 'contamination') and self.model.contamination is not None:
                    # Find threshold that would flag the top 'contamination' fraction as anomalies
                    threshold = np.percentile(scores, 100 * (1 - self.model.contamination))
                else:
                    # Default to 3 standard deviations from the mean
                    threshold = np.mean(scores) + 3 * np.std(scores)
        
        # Detect anomalies
        is_anomaly = scores > threshold
        
        return {
            'scores': scores,
            'is_anomaly': is_anomaly,
            'threshold': threshold
        }
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], 
                y_true: np.ndarray,
                threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Evaluate the anomaly detection performance.
        
        Args:
            X: Input features (DataFrame or numpy array)
            y_true: True labels (1 for anomaly, 0 for normal)
            threshold: Threshold for anomaly detection
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Detect anomalies
        result = self.detect_anomalies(X, threshold)
        y_pred = result['is_anomaly'].astype(int)
        
        # Calculate metrics
        metrics = {
            'threshold': result['threshold'],
            'num_anomalies': int(np.sum(y_pred)),
            'anomaly_ratio': float(np.mean(y_pred)),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Add ROC AUC if possible
        if len(np.unique(y_true)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, result['scores'])
            except ValueError:
                pass
        
        # Add confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        })
        
        return metrics
