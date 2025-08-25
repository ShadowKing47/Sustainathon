"""
Time Series Forecasting for Energy Consumption

This module provides a comprehensive framework for energy consumption forecasting
using various statistical and machine learning models, including Prophet, ARIMA,
LSTM, and Random Forest. It supports both univariate and multivariate forecasting,
model evaluation, and scenario analysis.
"""
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import warnings
from sklearn.metrics import (
    mean_absolute_error, 
    mean_absolute_percentage_error,
    mean_squared_error, 
    r2_score,
    explained_variance_score,
    median_absolute_error
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import config
from .config_loader import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Type aliases
ForecastResult = Dict[str, Union[np.ndarray, pd.DatetimeIndex]]
EvaluationMetrics = Dict[str, float]

# Custom exception for forecasting errors
class ForecastingError(Exception):
    """Custom exception for forecasting-related errors."""
    pass

class EnergyForecaster:
    """
    Forecast energy consumption using various statistical and machine learning models.
    
    Supported models:
    - 'prophet': Facebook's Prophet for time series forecasting
    - 'arima': AutoARIMA for automatic parameter selection
    - 'lstm': Long Short-Term Memory neural network
    - 'random_forest': Random Forest regressor
    """
    
    def __init__(self, model_type: str = 'prophet', **kwargs):
        """
        Initialize the EnergyForecaster with specified model type and parameters.
        
        Args:
            model_type: Type of forecasting model to use
            **kwargs: Additional model-specific parameters
        """
        self.model_type = model_type.lower()
        self.model = None
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.fitted = False
        self.params = kwargs
        self.last_training_date = None
        self.feature_columns = None
        self.target_column = None
        self.time_features = ['hour', 'dayofweek', 'month', 'is_weekend']
        self.training_metrics = None
        
        # Default parameters for each model type
        self.model_params = {
            'prophet': {
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': True,
                'seasonality_mode': 'multiplicative',
                'interval_width': 0.95,
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0
            },
            'arima': {
                'order': (1, 1, 1),
                'seasonal_order': (1, 1, 1, 24),
                'trend': 'c',
                'stepwise': True,
                'suppress_warnings': True,
                'error_action': 'ignore'
            },
            'lstm': {
                'units': 50,
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'epochs': 50,
                'batch_size': 32,
                'validation_split': 0.1,
                'verbose': 0
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'n_jobs': -1
            }
        }
        
        # Update with user params
        if self.model_type in self.model_params:
            self.model_params[self.model_type].update(kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(self.model_params.keys())}")
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features to the dataframe.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with added time features
        """
        df = df.copy()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        return df
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Prepare feature matrix with time features and lagged values.
        
        Args:
            df: Input DataFrame with target and features
            
        Returns:
            Tuple of (X, y, dates) where:
            - X: Feature matrix
            - y: Target values
            - dates: Corresponding dates
        """
        # Add time features
        df = self._add_time_features(df)
        
        # Add lagged values if needed
        if self.model_type in ['lstm', 'random_forest']:
            for lag in range(1, 4):  # Add 3 lags
                df[f'lag_{lag}'] = df[self.target_column].shift(lag)
        
        # Select features
        features = []
        
        # Add original features if specified
        if self.feature_columns:
            features.extend(self.feature_columns)
        
        # Add time features
        features.extend(self.time_features)
        
        # Add lagged features for certain models
        if self.model_type in ['lstm', 'random_forest']:
            features.extend([f'lag_{i}' for i in range(1, 4) if f'lag_{i}' in df.columns])
        
        # Drop rows with NaN values
        df = df[features + [self.target_column]].dropna()
        
        return df[features].values, df[self.target_column].values, df.index
    
    def prepare_data(self, data: pd.DataFrame, 
                    target_column: str,
                    timestamp_column: Optional[str] = None,
                    feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Prepare data for forecasting with feature engineering and scaling.
        
        Args:
            data: Input DataFrame with time series data
            target_column: Name of the target variable column
            timestamp_column: Name of the timestamp column (if not index)
            feature_columns: List of feature columns to use
            
        Returns:
            Dictionary containing prepared data and metadata
        """
        df = data.copy()
        
        # Set timestamp as index if specified
        if timestamp_column and timestamp_column in df.columns:
            df = df.set_index(timestamp_column)
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort and handle missing values
        df = df.sort_index().ffill().bfill()
        
        # Set target and features
        self.target_column = target_column
        self.feature_columns = feature_columns or [
            col for col in df.columns if col != target_column
        ]
        
        # Prepare features and target
        X, y, dates = self._prepare_features(df)
        
        # Scale features and target
        if len(X) > 0:
            X_scaled = self.scaler_features.fit_transform(X)
        else:
            X_scaled = np.array([])
            
        y_scaled = self.scaler_target.fit_transform(y.reshape(-1, 1)).flatten()
        
        return {
            'dates': dates,
            'X': X_scaled,
            'y': y_scaled,
            'feature_names': self.feature_columns + self.time_features
        }
    
    def fit(self, data: pd.DataFrame, 
            target_column: str,
            timestamp_column: Optional[str] = None,
            feature_columns: Optional[List[str]] = None,
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> 'EnergyForecaster':
        """
        Fit the forecasting model to the training data.
        
        Args:
            data: Training data as a pandas DataFrame
            target_column: Name of the target variable column
            timestamp_column: Name of the timestamp column (if not index)
            feature_columns: List of feature columns to use
            validation_data: Optional tuple of (X_val, y_val) for validation
            
        Returns:
            self: Returns the instance itself
        """
        # Prepare training data
        prepared_data = self.prepare_data(
            data, target_column, timestamp_column, feature_columns
        )
        
        # Fit the appropriate model
        if self.model_type == 'prophet':
            self._fit_prophet(prepared_data)
        elif self.model_type == 'arima':
            self._fit_arima(prepared_data)
        elif self.model_type == 'lstm':
            self._fit_lstm(prepared_data, validation_data)
        elif self.model_type == 'random_forest':
            self._fit_random_forest(prepared_data, validation_data)
        
        self.fitted = True
        self.last_training_date = prepared_data['dates'][-1]
        
        # Calculate training metrics
        self.training_metrics = self._calculate_training_metrics(prepared_data)
        
        return self
        
    def _fit_prophet(self, data: Dict[str, Any]) -> None:
        """
        Fit Facebook's Prophet model to the data.
        
        Args:
            data: Dictionary containing prepared data
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet not installed. Run: pip install prophet")
        
        # Prepare DataFrame for Prophet
        df = pd.DataFrame({
            'ds': data['dates'],
            'y': data['y']
        })
        
        # Add features if available
        if data['X'] is not None and len(data['X']) > 0:
            for i, col in enumerate(self.feature_columns):
                df[col] = data['X'][:, i]
        
        # Initialize and fit model
        self.model = Prophet(**self.model_params['prophet'])
        
        # Add regressors
        if data['X'] is not None and len(data['X']) > 0:
            for col in self.feature_columns:
                self.model.add_regressor(col)
        
        # Add holiday effects if specified
        if self.params.get('country_holidays'):
            self.model.add_country_holidays(country_name=self.params['country_holidays'])
        
        # Add custom seasonalities if specified
        if 'custom_seasonalities' in self.params:
            for seasonality in self.params['custom_seasonalities']:
                self.model.add_seasonality(**seasonality)
        
        self.model.fit(df)
        
        # Store feature importance
        if hasattr(self.model, 'component_modes'):
            self.feature_importance_ = self.model.component_modes
    
    def _fit_arima(self, data: Dict[str, Any]) -> None:
        """
        Fit AutoARIMA model to the data.
        
        Args:
            data: Dictionary containing prepared data
        """
        try:
            from pmdarima import auto_arima
        except ImportError:
            raise ImportError("pmdarima not installed. Run: pip install pmdarima")
        
        try:
            self.model = auto_arima(
                y=data['y'],
                X=data['X'] if 'X' in data and len(data['X']) > 0 else None,
                **self.model_params['arima']
            )
            
            # Store model summary
            self.model_summary_ = self.model.summary()
            
        except Exception as e:
            raise ForecastingError(f"Error fitting ARIMA model: {str(e)}")
    
    def _fit_lstm(self, data: Dict[str, Any], validation_data: Optional[Tuple] = None) -> None:
        """
        Fit LSTM model to the data.
        
        Args:
            data: Dictionary containing prepared data
            validation_data: Optional tuple of (X_val, y_val) for validation
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")
        
        # Reshape data for LSTM [samples, timesteps, features]
        X = data['X']
        y = data['y']
        
        # Reshape input to be 3D [samples, timesteps, features]
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        X_reshaped = X.reshape((X.shape[0], 1, n_features))
        
        # Define model
        model = Sequential()
        model.add(LSTM(
            units=self.model_params['lstm']['units'],
            input_shape=(1, n_features),
            return_sequences=True,
            dropout=self.model_params['lstm']['dropout'],
            recurrent_dropout=self.model_params['lstm']['recurrent_dropout']
        ))
        model.add(LSTM(
            units=self.model_params['lstm']['units'] // 2,
            dropout=self.model_params['lstm']['dropout'],
            recurrent_dropout=self.model_params['lstm']['recurrent_dropout']
        ))
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Prepare validation data
        validation = None
        if validation_data:
            X_val, y_val = validation_data
            X_val_reshaped = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
            validation = (X_val_reshaped, y_val)
        
        # Train model
        history = model.fit(
            X_reshaped, y,
            epochs=self.model_params['lstm']['epochs'],
            batch_size=self.model_params['lstm']['batch_size'],
            validation_split=self.model_params['lstm']['validation_split'],
            validation_data=validation,
            callbacks=callbacks,
            verbose=self.model_params['lstm']['verbose']
        )
        
        self.model = model
        self.training_history_ = history.history
    
    def _fit_random_forest(self, data: Dict[str, Any], validation_data: Optional[Tuple] = None) -> None:
        """
        Fit Random Forest model to the data.
        
        Args:
            data: Dictionary containing prepared data
            validation_data: Optional tuple of (X_val, y_val) for validation
        """
        from sklearn.ensemble import RandomForestRegressor
        
        X = data['X']
        y = data['y']
        
        # Initialize and fit model
        self.model = RandomForestRegressor(
            **self.model_params['random_forest']
        )
        
        self.model.fit(X, y)
        
        # Store feature importances
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_,
                index=data.get('feature_names', range(X.shape[1]))
            ).sort_values(ascending=False)
    
    def _calculate_training_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate training metrics for the fitted model.
        
        Args:
            data: Dictionary containing prepared data
            
        Returns:
            Dictionary of training metrics
        """
        if not self.fitted or self.model is None:
            return {}
        
        X = data['X']
        y_true = data['y']
        
        # Get predictions
        if self.model_type == 'prophet':
            preds = self.model.predict(pd.DataFrame({'ds': data['dates']}))['yhat'].values
        elif self.model_type == 'arima':
            preds = self.model.predict_in_sample(X) if hasattr(self.model, 'predict_in_sample') else None
        elif self.model_type in ['lstm', 'random_forest']:
            preds = self.model.predict(X)
        else:
            preds = None
        
        if preds is None or len(preds) == 0:
            return {}
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_true, preds),
            'rmse': np.sqrt(mean_squared_error(y_true, preds)),
            'r2': r2_score(y_true, preds),
            'mape': mean_absolute_percentage_error(y_true, preds) * 100,
            'explained_variance': explained_variance_score(y_true, preds),
            'median_ae': median_absolute_error(y_true, preds)
        }
        
        return metrics
    
    def _fit_prophet(self, data: Dict[str, Any]) -> None:
        """Fit Prophet model."""
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet not installed. Run: pip install prophet")
        
        # Prepare DataFrame for Prophet
        df = pd.DataFrame({
            'ds': data['dates'],
            'y': data['values']
        })
        
        # Add features if available
        if data['features'] is not None:
            for i, col in enumerate(self.feature_columns):
                df[col] = data['features'][:, i]
        
        # Initialize and fit model
        self.model = Prophet(**self.model_params['prophet'])
        
        # Add regressors
        if data['features'] is not None:
            for col in self.feature_columns:
                self.model.add_regressor(col)
        
        self.model.fit(df)
    
    def _fit_arima(self, data: Dict[str, Any]) -> None:
        """Fit ARIMA model."""
        try:
            from pmdarima import auto_arima
        except ImportError:
            raise ImportError("pmdarima not installed. Run: pip install pmdarima")
        
        self.model = auto_arima(
            data['values'],
            X=data['features'],
            **self.model_params['arima']
        )
    
    def predict(self, n_periods: int,
                X: Optional[np.ndarray] = None,
                return_conf_int: bool = True) -> ForecastResult:
        """
        Generate forecasts for the specified number of periods.
        
        Args:
            n_periods: Number of periods to forecast
            X: Exogenous variables for the forecast period
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Dictionary containing forecast values and confidence intervals
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        try:
            if self.model_type == 'prophet':
                return self._predict_prophet(n_periods, X, return_conf_int)
            elif self.model_type == 'arima':
                return self._predict_arima(n_periods, X, return_conf_int)
            elif self.model_type == 'lstm':
                return self._predict_lstm(n_periods, X, return_conf_int)
            elif self.model_type == 'random_forest':
                return self._predict_random_forest(n_periods, X, return_conf_int)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            raise ForecastingError(f"Error generating forecasts: {str(e)}")
            
    def _prepare_forecast_dates(self, n_periods: int) -> pd.DatetimeIndex:
        """Generate future dates for forecasting."""
        freq = pd.infer_freq(pd.DatetimeIndex([
            self.last_training_date - pd.Timedelta(days=1), 
            self.last_training_date
        ]))
        
        return pd.date_range(
            start=self.last_training_date + pd.Timedelta(1, freq[0] if freq else 'D'),
            periods=n_periods,
            freq=freq if freq else 'D'
        )
    
    def _predict_prophet(self, n_periods: int, 
                        X: Optional[np.ndarray] = None,
                        return_conf_int: bool = True) -> ForecastResult:
        """
        Generate forecasts using Facebook's Prophet model.
        
        Args:
            n_periods: Number of periods to forecast
            X: Exogenous variables for the forecast period
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Dictionary containing forecast results
        """
        # Create future dates
        future_dates = self._prepare_forecast_dates(n_periods)
        
        # Create future DataFrame
        future = pd.DataFrame({'ds': future_dates})
        
        # Add features if available
        if X is not None and self.feature_columns:
            for i, col in enumerate(self.feature_columns):
                future[col] = X[:, i] if len(X.shape) > 1 else X
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        # Inverse transform if scaled
        forecast_values = forecast['yhat'].values
        if hasattr(self, 'scaler_target'):
            forecast_values = self.scaler_target.inverse_transform(
                forecast_values.reshape(-1, 1)
            ).flatten()
        
        result = {
            'forecast': forecast_values,
            'dates': future_dates
        }
        
        # Add confidence intervals if requested
        if return_conf_int and 'yhat_lower' in forecast.columns:
            if hasattr(self, 'scaler_target'):
                lower = self.scaler_target.inverse_transform(
                    forecast['yhat_lower'].values.reshape(-1, 1)
                ).flatten()
                upper = self.scaler_target.inverse_transform(
                    forecast['yhat_upper'].values.reshape(-1, 1)
                ).flatten()
            else:
                lower = forecast['yhat_lower'].values
                upper = forecast['yhat_upper'].values
                
            result.update({
                'lower': lower,
                'upper': upper
            })
        
        return result
    
    def _predict_arima(self, n_periods: int, 
                      X: Optional[np.ndarray] = None,
                      return_conf_int: bool = True) -> ForecastResult:
        """
        Generate forecasts using ARIMA model.
        
        Args:
            n_periods: Number of periods to forecast
            X: Exogenous variables for the forecast period
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Dictionary containing forecast results
        """
        try:
            # Generate forecast with confidence intervals
            if return_conf_int:
                forecast, conf_int = self.model.predict(
                    n_periods=n_periods,
                    X=X,
                    return_conf_int=True
                )
                
                # Inverse transform if scaled
                if hasattr(self, 'scaler_target'):
                    forecast_values = self.scaler_target.inverse_transform(
                        forecast.values.reshape(-1, 1)
                    ).flatten()
                    lower = self.scaler_target.inverse_transform(
                        conf_int[:, 0].reshape(-1, 1)
                    ).flatten()
                    upper = self.scaler_target.inverse_transform(
                        conf_int[:, 1].reshape(-1, 1)
                    ).flatten()
                else:
                    forecast_values = forecast.values
                    lower = conf_int[:, 0]
                    upper = conf_int[:, 1]
                
                result = {
                    'forecast': forecast_values,
                    'lower': lower,
                    'upper': upper,
                    'dates': self._prepare_forecast_dates(n_periods)
                }
            else:
                forecast = self.model.predict(
                    n_periods=n_periods,
                    X=X,
                    return_conf_int=False
                )
                
                # Inverse transform if scaled
                if hasattr(self, 'scaler_target'):
                    forecast_values = self.scaler_target.inverse_transform(
                        forecast.values.reshape(-1, 1)
                    ).flatten()
                else:
                    forecast_values = forecast.values
                
                result = {
                    'forecast': forecast_values,
                    'dates': self._prepare_forecast_dates(n_periods)
                }
            
            return result
            
        except Exception as e:
            raise ForecastingError(f"Error in ARIMA prediction: {str(e)}")
            
    def _predict_lstm(self, n_periods: int,
                     X: Optional[np.ndarray] = None,
                     return_conf_int: bool = True) -> ForecastResult:
        """
        Generate forecasts using LSTM model.
        
        Args:
            n_periods: Number of periods to forecast
            X: Exogenous variables for the forecast period
            return_conf_int: Whether to return confidence intervals (not fully supported for LSTM)
            
        Returns:
            Dictionary containing forecast results
        """
        if X is None or len(X) < n_periods:
            raise ValueError("Exogenous variables are required for LSTM forecasting")
        
        # Prepare input data
        X_forecast = X[:n_periods]
        
        # Reshape for LSTM [samples, timesteps, features]
        n_features = X_forecast.shape[1] if len(X_forecast.shape) > 1 else 1
        X_reshaped = X_forecast.reshape((X_forecast.shape[0], 1, n_features))
        
        # Generate predictions
        preds = self.model.predict(X_reshaped).flatten()
        
        # Inverse transform if scaled
        if hasattr(self, 'scaler_target'):
            preds = self.scaler_target.inverse_transform(preds.reshape(-1, 1)).flatten()
        
        result = {
            'forecast': preds,
            'dates': self._prepare_forecast_dates(n_periods)
        }
        
        # Add placeholder confidence intervals if requested
        if return_conf_int:
            result.update({
                'lower': preds * 0.95,  # Placeholder
                'upper': preds * 1.05   # Placeholder
            })
            
        return result
        
    def _predict_random_forest(self, n_periods: int,
                             X: Optional[np.ndarray] = None,
                             return_conf_int: bool = True) -> ForecastResult:
        """
        Generate forecasts using Random Forest model.
        
        Args:
            n_periods: Number of periods to forecast
            X: Exogenous variables for the forecast period
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Dictionary containing forecast results
        """
        if X is None or len(X) < n_periods:
            raise ValueError("Exogenous variables are required for Random Forest forecasting")
        
        # Prepare input data
        X_forecast = X[:n_periods]
        
        # Generate predictions
        preds = self.model.predict(X_forecast)
        
        # Inverse transform if scaled
        if hasattr(self, 'scaler_target'):
            preds = self.scaler_target.inverse_transform(preds.reshape(-1, 1)).flatten()
        
        result = {
            'forecast': preds,
            'dates': self._prepare_forecast_dates(n_periods)
        }
        
        # Add confidence intervals if requested
        if return_conf_int and hasattr(self.model, 'estimators_'):
            # Calculate prediction intervals using the ensemble
            predictions = np.column_stack([
                tree.predict(X_forecast) for tree in self.model.estimators_
            ])
            
            if hasattr(self, 'scaler_target'):
                predictions = np.column_stack([
                    self.scaler_target.inverse_transform(p.reshape(-1, 1)).flatten()
                    for p in predictions.T
                ])
            
            std = np.std(predictions, axis=1)
            result.update({
                'lower': preds - 1.96 * std,
                'upper': preds + 1.96 * std
            })
            
        return result
    
    def evaluate(self, y_true: np.ndarray, 
                y_pred: np.ndarray,
                sample_weight: Optional[np.ndarray] = None) -> EvaluationMetrics:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            sample_weight: Optional array of weights for weighted metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred, sample_weight=sample_weight),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)),
            'r2': r2_score(y_true, y_pred, sample_weight=sample_weight),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'explained_variance': explained_variance_score(y_true, y_pred, sample_weight=sample_weight),
            'median_ae': median_absolute_error(y_true, y_pred)
        }
        
        # Add additional metrics if needed
        try:
            from scipy.stats import pearsonr, spearmanr
            corr = pearsonr(y_true, y_pred)[0]
            metrics['pearson_r'] = corr
            metrics['spearman_rho'] = spearmanr(y_true, y_pred)[0]
        except ImportError:
            pass
            
        return metrics
        
    def cross_validate(self, 
                      data: pd.DataFrame,
                      target_column: str,
                      n_splits: int = 5,
                      test_size: Optional[float] = 0.2,
                      gap: int = 0,
                      **fit_params) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            n_splits: Number of splits for cross-validation
            test_size: Fraction of data to use for testing in each split
            gap: Number of samples to exclude from the end of each training set
            **fit_params: Additional parameters to pass to the fit method
            
        Returns:
            Dictionary of evaluation metrics across all folds
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        # Prepare data
        X = data.drop(columns=[target_column])
        y = data[target_column].values
        
        # Initialize cross-validator
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap
        )
        
        # Initialize results
        metrics = {
            'mae': [],
            'rmse': [],
            'r2': [],
            'mape': []
        }
        
        # Perform cross-validation
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Fit model
            self.fit(
                X_train, 
                target_column,
                **fit_params
            )
            
            # Make predictions
            y_pred = self.predict(len(X_test), X_test)['forecast']
            
            # Calculate metrics
            fold_metrics = self.evaluate(y_test, y_pred)
            
            # Store results
            for metric in metrics:
                if metric in fold_metrics:
                    metrics[metric].append(fold_metrics[metric])
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model and its configuration to a file.
        
        Args:
            filepath: Path to save the model file
            
        Raises:
            RuntimeError: If the model is not fitted
        """
        import joblib
        import logging
        
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        model_data = {
            'model_type': self.model_type,
            'model': self.model,
            'params': self.params,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'time_features': self.time_features,
            'last_training_date': self.last_training_date,
            'training_metrics': self.training_metrics,
            'model_params': self.model_params,
            'fitted': self.fitted,
            'scaler_features': self.scaler_features,
            'scaler_target': self.scaler_target
        }
        
        # Save additional model-specific data
        if hasattr(self, 'feature_importance_'):
            model_data['feature_importance'] = self.feature_importance_
        if hasattr(self, 'training_history_'):
            model_data['training_history'] = self.training_history_
        logger = logging.getLogger(__name__)
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved successfully to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EnergyForecaster':
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Loaded EnergyForecaster instance
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
            RuntimeError: If there's an error loading the model
        """
        try:
            import joblib
            import logging
            logger = logging.getLogger(__name__)
            model_data = joblib.load(filepath)
            
            # Create new instance
            forecaster = cls(
                model_type=model_data['model_type'],
                **model_data['params']
            )
            
            # Set attributes
            forecaster.model = model_data['model']
            forecaster.feature_columns = model_data.get('feature_columns')
            forecaster.target_column = model_data.get('target_column')
            forecaster.time_features = model_data.get('time_features', [])
            forecaster.last_training_date = model_data.get('last_training_date')
            forecaster.training_metrics = model_data.get('training_metrics')
            forecaster.model_params = model_data.get('model_params', {})
            forecaster.fitted = model_data.get('fitted', False)
            forecaster.scaler_features = model_data.get('scaler_features', StandardScaler())
            forecaster.scaler_target = model_data.get('scaler_target', StandardScaler())
            
            # Set additional attributes if they exist
            if 'feature_importance' in model_data:
                forecaster.feature_importance_ = model_data['feature_importance']
            if 'training_history' in model_data:
                forecaster.training_history_ = model_data['training_history']
            
            logger.info(f"Model loaded successfully from {filepath}")
            return forecaster
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    @classmethod
    def plot_forecast(cls, 
                      forecast: Dict[str, np.ndarray],
                      actual: Optional[Dict[str, np.ndarray]] = None,
                      title: str = 'Energy Consumption Forecast',
                      figsize: Tuple[int, int] = (12, 6),
                      **kwargs) -> plt.Figure:
        """
        Plot the forecasted values with confidence intervals.
        
        Args:
            forecast: Dictionary containing 'forecast', 'dates', and optionally 'lower'/'upper' keys
            actual: Optional dictionary with 'values' and 'dates' for actual data
            title: Plot title
            figsize: Figure size (width, height)
            **kwargs: Additional arguments to pass to the plot
            
        Returns:
            Matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        
        # Plot actual data if provided
        if actual is not None:
            plt.plot(
                actual['dates'], 
                actual['values'], 
                label='Actual',
                color='#1f77b4',
                alpha=0.8
            )
        
        # Plot forecast
        forecast_dates = forecast['dates']
        forecast_values = forecast['forecast']
        
        # Plot confidence interval if available
        if 'lower' in forecast and 'upper' in forecast:
            plt.fill_between(
                forecast_dates,
                forecast['lower'],
                forecast['upper'],
                color='#1f77b4',
                alpha=0.2,
                label='Confidence Interval'
            )
        
        # Plot forecast line
        plt.plot(
            forecast_dates,
            forecast_values,
            label='Forecast',
            color='#ff7f0e',
            linewidth=2
        )
        
        # Add vertical line at forecast start if actual data is provided
        if actual is not None and len(actual['dates']) > 0:
            forecast_start = min(forecast_dates)
            plt.axvline(
                x=forecast_start,
                color='red',
                linestyle='--',
                label='Forecast Start'
            )
        
        # Customize plot
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Energy Consumption', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_feature_importance(self, 
                              top_n: Optional[int] = 10,
                              figsize: Tuple[int, int] = (10, 6),
                              **kwargs) -> Optional[plt.Figure]:
        """
        Plot feature importance if available for the model.
        
        Args:
            top_n: Number of top features to show
            figsize: Figure size (width, height)
            **kwargs: Additional arguments to pass to the plot
            
        Returns:
            Matplotlib Figure object if feature importance is available, else None
        """
        import matplotlib.pyplot as plt
        if not hasattr(self, 'feature_importance_') or self.feature_importance_ is None:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Feature importance not available for this model")
            return None
            
        # Get feature importance data
        if isinstance(self.feature_importance_, pd.Series):
            importance = self.feature_importance_
        else:
            # Handle other formats if needed
            importance = pd.Series(self.feature_importance_)
        
        # Sort and select top N features
        importance = importance.sort_values(ascending=False)
        if top_n is not None and len(importance) > top_n:
            importance = importance.head(top_n)
        
        # Create plot
        plt.figure(figsize=figsize)
        importance.plot(kind='barh', color='skyblue', edgecolor='black')
        
        # Customize plot
        plt.title('Feature Importance', fontsize=14, pad=20)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_training_history(self, 
                            metrics: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (12, 6),
                            **kwargs) -> Optional['plt.Figure']:
        """
        Plot training history for models that support it (e.g., LSTM).
        
        Args:
            metrics: List of metrics to plot (e.g., ['loss', 'val_loss'])
            figsize: Figure size (width, height)
            **kwargs: Additional arguments to pass to the plot
            
        Returns:
            Matplotlib Figure object if training history is available, else None
        """
        import matplotlib.pyplot as plt
        if not hasattr(self, 'training_history_') or not self.training_history_:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Training history not available for this model")
            return None
            
        history = self.training_history_
        
        # Determine metrics to plot
        if metrics is None:
            metrics = [key for key in history.keys() if not key.startswith('val_')]
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Plot training metric
            ax.plot(history[metric], label=f'Training {metric}')
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history:
                ax.plot(history[val_metric], label=f'Validation {metric}')
            
            # Customize subplot
            ax.set_title(f'{metric.title()} Over Epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.title())
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance if available for the model.
        
        Returns:
            Pandas Series with feature importance if available, else None
        """
        if hasattr(self, 'feature_importance_'):
            if isinstance(self.feature_importance_, pd.Series):
                return self.feature_importance_
            return pd.Series(
                self.feature_importance_,
                index=getattr(self, 'feature_columns', range(len(self.feature_importance_))))
        return None
    
    def get_model_summary(self) -> Optional[str]:
        """
        Get a summary of the model if available.
        
        Returns:
            Model summary as a string if available, else None
        """
        if hasattr(self, 'model_summary_'):
            return str(self.model_summary_)
        return None
