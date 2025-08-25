"""
Data preprocessing utilities for the Energy Monitoring System.
Handles data cleaning, transformation, and feature engineering for energy data analysis.
"""
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import holidays
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, FunctionTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import yaml

# Import config
from .config_loader import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data preprocessing tasks for the energy monitoring system."""
    
    def __init__(self, dataset_name: str = None, config_path: str = None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            dataset_name: Name of the dataset (optional, used for schema lookups)
            config_path: Path to config file (if not using default config_loader)
        """
        self.dataset_name = dataset_name
        self.config = self._load_config(config_path) if config_path else config
        self.schema = self.config.get_schema(dataset_name) if dataset_name else {}
        
        # Column specifications
        self.feature_columns = self.schema.get('features', [])
        self.target_columns = self.schema.get('targets', ['consumption'])
        self.datetime_columns = self.schema.get('datetime_columns', ['timestamp'])
        self.categorical_columns = self.schema.get('categorical_columns', [])
        self.numeric_columns = self.schema.get('numeric_columns', [])
        self.derived_metrics = self.schema.get('derived_metrics', {})
        
        # Preprocessing parameters
        self.timezone = pytz.timezone(self.schema.get('timezone', 'UTC'))
        self.outlier_threshold = self.schema.get('outlier_threshold', 3.0)
        self.imputation_strategy = self.schema.get('imputation_strategy', 'interpolate')
        
        # Initialize transformers and state
        self.preprocessor = None
        self.scaler = None
        self.imputer = None
        self.feature_engineering_pipeline = None
        self.fitted = False
    
    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from file if provided, otherwise use default."""
        if config_path:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return config

    def load_data(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats with automatic detection.
        
        Supported formats: CSV, Excel, Parquet, JSON, SQL
        """
        """
        Load data from a file with automatic format detection.
        
        Args:
            filepath: Path to the data file
            **kwargs: Additional arguments to pass to the loader
            
        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Determine file type from extension
        ext = filepath.suffix.lower()
        
        try:
            if ext == '.csv':
                # Try different encodings for CSV
                encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding, **kwargs)
                        logger.info(f"Successfully loaded {filepath} with {encoding} encoding")
                        return self._post_load(df)
                    except UnicodeDecodeError:
                        continue
                raise ValueError(f"Failed to load {filepath} with any supported encoding")
            
            elif ext == '.parquet':
                df = pd.read_parquet(filepath, **kwargs)
                return self._post_load(df)
            
            elif ext in ['.xls', '.xlsx']:
                df = pd.read_excel(filepath, **kwargs)
                return self._post_load(df)
            
            elif ext == '.json':
                df = pd.read_json(filepath, **kwargs)
                return self._post_load(df)
            
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise
    
    def _post_load(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply post-loading transformations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        # Convert column names to lowercase with underscores
        df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
        
        # Apply schema-based type conversions
        if self.schema and 'dtypes' in self.schema:
            for col, dtype in self.schema['dtypes'].items():
                if col in df.columns:
                    if dtype == 'datetime':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif dtype == 'numeric':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def detect_datetime_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect datetime columns in a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of datetime column names
        """
        datetime_cols = []
        
        for col in df.columns:
            # Check if column name suggests it's a datetime
            if any(term in col.lower() for term in ['date', 'time', 'timestamp']):
                datetime_cols.append(col)
                continue
            
            # Check if column values can be converted to datetime
            try:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_cols.append(col)
                elif df[col].dropna().shape[0] > 0:  # If not empty after dropping NA
                    sample = df[col].dropna().sample(min(100, len(df[col].dropna())))
                    pd.to_datetime(sample, errors='raise')
                    datetime_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        return datetime_cols
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features to the DataFrame.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with added time features
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            dt_col = self.datetime_columns[0] if self.datetime_columns else 'timestamp'
            df[dt_col] = pd.to_datetime(df[dt_col])
            df = df.set_index(dt_col)
        
        # Basic time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        
        # Add holiday information
        country_holidays = holidays.CountryHoliday(self.schema.get('country', 'US'))
        df['is_holiday'] = df.index.date.astype('datetime64[ns]').isin(
            pd.to_datetime(list(country_holidays.years[df.index.year.min():df.index.year.max() + 1]))
        ).astype(int)
        
        # Add season
        df['season'] = (df.index.month % 12 + 3) // 3
        
        # Add time of day categories
        bins = [0, 6, 12, 18, 23]
        labels = ['night', 'morning', 'afternoon', 'evening']
        df['time_of_day'] = pd.cut(df.index.hour, bins=bins, labels=labels, include_lowest=True)
        
        return df
    
    def _add_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived metrics based on configuration."""
        for metric, formula in self.derived_metrics.items():
            try:
                df[metric] = df.eval(formula)
            except Exception as e:
                logger.warning(f"Could not calculate metric {metric}: {str(e)}")
        return df
    
    def detect_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect numeric columns in a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of numeric column names
        """
        numeric_cols = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                try:
                    # Try to convert to numeric
                    pd.to_numeric(df[col], errors='raise')
                    numeric_cols.append(col)
                except (ValueError, TypeError):
                    continue
        
        return numeric_cols
    
    def detect_categorical_columns(self, df: pd.DataFrame, max_categories: int = 20) -> List[str]:
        """
        Detect categorical columns in a DataFrame.
        
        Args:
            df: Input DataFrame
            max_categories: Maximum number of unique values to consider a column categorical
            
        Returns:
            List of categorical column names
        """
        categorical_cols = []
        
        for col in df.columns:
            # Skip datetime columns
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
                
            # Check if column has a reasonable number of unique values
            unique_count = df[col].nunique()
            if 1 < unique_count <= max_categories:
                categorical_cols.append(col)
            # Also consider string columns with low cardinality
            elif pd.api.types.is_string_dtype(df[col]) and unique_count <= max_categories:
                categorical_cols.append(col)
        
        return categorical_cols
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None, method: str = 'zscore', 
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in numeric columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers (default: all numeric columns)
            method: Method for outlier detection ('zscore' or 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with boolean columns indicating outliers
        """
        if columns is None:
            columns = self.detect_numeric_columns(df)
        
        outliers = pd.DataFrame(index=df.index)
        
        for col in columns:
            if method == 'zscore':
                z_scores = (df[col] - df[col].mean()) / df[col].std()
                outliers[f'{col}_outlier'] = z_scores.abs() > threshold
            elif method == 'iqr':
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outliers[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        return outliers
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = None, 
                            numeric_columns: List[str] = None, 
                            categorical_columns: List[str] = None) -> pd.DataFrame:
        """
        Handle missing values using various strategies.
        
        Args:
            df: Input DataFrame
            strategy: One of 'mean', 'median', 'mode', 'interpolate', 'knn', 'drop'
            numeric_columns: List of numeric column names
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with missing values handled
        """
        strategy = strategy or self.imputation_strategy
        df = df.copy()
        
        # Handle numeric columns
        num_cols = numeric_columns or self.numeric_columns or df.select_dtypes(include=np.number).columns.tolist()
        
        if strategy == 'interpolate':
            df[num_cols] = df[num_cols].interpolate(method='time')
            df[num_cols] = df[num_cols].fillna(method='ffill')
            df[num_cols] = df[num_cols].fillna(method='bfill')
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            df[num_cols] = imputer.fit_transform(df[num_cols])
        elif strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
            df[num_cols] = imputer.fit_transform(df[num_cols])
        elif strategy == 'drop':
            df = df.dropna(subset=num_cols)
            
        # Handle categorical columns
        cat_cols = categorical_columns or self.categorical_columns or df.select_dtypes(include=['object', 'category']).columns.tolist()
        df[cat_cols] = df[cat_cols].fillna('missing')
        
        return df
        """
        Handle missing values in a DataFrame.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for imputation ('mean', 'median', 'most_frequent', 'constant')
            numeric_columns: List of numeric columns to impute
            categorical_columns: List of categorical columns to impute
            
        Returns:
            DataFrame with missing values imputed
        """
        df = df.copy()
        
        if numeric_columns is None:
            numeric_columns = self.detect_numeric_columns(df)
        
        if categorical_columns is None:
            categorical_columns = self.detect_categorical_columns(df)
        
        # Impute numeric columns
        for col in numeric_columns:
            if df[col].isna().any():
                if strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == 'most_frequent':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif strategy == 'constant':
                    df[col].fillna(0, inplace=True)
        
        # Impute categorical columns
        for col in categorical_columns:
            if df[col].isna().any():
                if strategy in ['most_frequent', 'constant']:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna('missing', inplace=True)
        
        return df
    
    def create_features(self, df: pd.DataFrame, datetime_column: str = None) -> pd.DataFrame:
        """
        Create time-based features from a datetime column.
        
        Args:
            df: Input DataFrame
            datetime_column: Name of the datetime column
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        if datetime_column is None:
            datetime_cols = self.detect_datetime_columns(df)
            if datetime_cols:
                datetime_column = datetime_cols[0]
            else:
                logger.warning("No datetime column found for feature creation")
                return df
        
        if datetime_column not in df.columns:
            raise ValueError(f"Datetime column '{datetime_column}' not found in DataFrame")
        
        # Ensure the column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
            df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
        
        # Extract time-based features
        df[f'{datetime_column}_year'] = df[datetime_column].dt.year
        df[f'{datetime_column}_month'] = df[datetime_column].dt.month
        df[f'{datetime_column}_day'] = df[datetime_column].dt.day
        df[f'{datetime_column}_hour'] = df[datetime_column].dt.hour
        df[f'{datetime_column}_minute'] = df[datetime_column].dt.minute
        df[f'{datetime_column}_dayofweek'] = df[datetime_column].dt.dayofweek
        df[f'{datetime_column}_dayofyear'] = df[datetime_column].dt.dayofyear
        df[f'{datetime_column}_weekofyear'] = df[datetime_column].dt.isocalendar().week
        df[f'{datetime_column}_quarter'] = df[datetime_column].dt.quarter
        df[f'{datetime_column}_is_month_start'] = df[datetime_column].dt.is_month_start.astype(int)
        df[f'{datetime_column}_is_month_end'] = df[datetime_column].dt.is_month_end.astype(int)
        df[f'{datetime_column}_is_weekend'] = (df[datetime_column].dt.dayofweek >= 5).astype(int)
        
        # Time since reference (useful for trend detection)
        if len(df) > 1:
            min_date = df[datetime_column].min()
            df[f'{datetime_column}_time_since_start'] = (df[datetime_column] - min_date).dt.total_seconds() / 3600  # Hours
        
        return df
    
    def detect_and_remove_outliers(self, df: pd.DataFrame, columns: List[str] = None, 
                                 method: str = 'zscore', threshold: float = None) -> pd.DataFrame:
        """
        Detect and remove outliers using specified method.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: One of 'zscore', 'iqr', 'isolation_forest'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        if not columns:
            columns = self.numeric_columns or df.select_dtypes(include=np.number).columns.tolist()
            
        threshold = threshold or self.outlier_threshold
        df_clean = df.copy()
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(df_clean[columns]))
            mask = (z_scores < threshold).all(axis=1)
        elif method == 'iqr':
            Q1 = df_clean[columns].quantile(0.25)
            Q3 = df_clean[columns].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((df_clean[columns] < (Q1 - threshold * IQR)) | 
                    (df_clean[columns] > (Q3 + threshold * IQR))).any(axis=1)
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=0.1, random_state=42)
            mask = clf.fit_predict(df_clean[columns]) == 1
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
            
        logger.info(f"Removed {len(df) - sum(mask)} outliers using {method} method")
        return df_clean[mask]
    
    def create_lag_features(self, df: pd.DataFrame, value_column: str = None, 
                          datetime_column: str = None, 
                          lags: List[int] = None,
                          rolling_windows: List[int] = None) -> pd.DataFrame:
        """
        Create lagged features for time series data.
        
        Args:
            df: Input DataFrame
            value_column: Name of the column to create lags for
            datetime_column: Name of the datetime column
            lags: List of lag periods to create
            
        Returns:
            DataFrame with added lag features
        """
        if lags is None:
            lags = [1, 2, 3, 7, 14, 30]  # Common lags for daily data
        
        if datetime_column is not None and datetime_column in df.columns:
            # Sort by datetime to ensure correct lagging
            df = df.sort_values(datetime_column).copy()
        
        for lag in lags:
            df[f'{value_column}_lag_{lag}'] = df[value_column].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, value_column: str, 
                               windows: List[int] = None, 
                               aggregations: List[str] = None) -> pd.DataFrame:
        """
        Create rolling window features for time series data.
        
        Args:
            df: Input DataFrame
            value_column: Name of the column to create rolling features for
            windows: List of window sizes
            aggregations: List of aggregation functions ('mean', 'std', 'min', 'max', 'sum')
            
        Returns:
            DataFrame with added rolling features
        """
        if windows is None:
            windows = [3, 7, 14, 30]  # Common window sizes for daily data
        
        if aggregations is None:
            aggregations = ['mean', 'std', 'min', 'max']
        
        for window in windows:
            for agg in aggregations:
                if agg == 'mean':
                    df[f'{value_column}_rolling_{window}_{agg}'] = df[value_column].rolling(window=window).mean()
                elif agg == 'std':
                    df[f'{value_column}_rolling_{window}_{agg}'] = df[value_column].rolling(window=window).std()
                elif agg == 'min':
                    df[f'{value_column}_rolling_{window}_{agg}'] = df[value_column].rolling(window=window).min()
                elif agg == 'max':
                    df[f'{value_column}_rolling_{window}_{agg}'] = df[value_column].rolling(window=window).max()
                elif agg == 'sum':
                    df[f'{value_column}_rolling_{window}_{agg}'] = df[value_column].rolling(window=window).sum()
        
        return df
    
    def fit_preprocessor(self, X: pd.DataFrame, numeric_columns: List[str] = None, 
                        categorical_columns: List[str] = None):
        """
        Fit a preprocessing pipeline on the data.
        
        Args:
            X: Input features
            numeric_columns: List of numeric column names
            categorical_columns: List of categorical column names
        """
        if numeric_columns is None:
            numeric_columns = self.detect_numeric_columns(X)
        
        if categorical_columns is None:
            categorical_columns = self.detect_categorical_columns(X)
        
        # Define transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_columns),
                ('cat', categorical_transformer, categorical_columns)
            ])
        
        # Fit the preprocessor
        self.preprocessor.fit(X)
        self.fitted = True
    
    def transform_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted preprocessor.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features as a numpy array
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit_preprocessor first.")
        
        return self.preprocessor.transform(X)
    
    def create_feature_pipeline(self) -> Pipeline:
        """
        Create a feature engineering pipeline.
        
        Returns:
            Configured scikit-learn Pipeline
        """
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.imputation_strategy)),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_columns),
                ('cat', categorical_transformer, self.categorical_columns)
            ])
            
        self.feature_engineering_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_union', FeatureUnion([
                ('time_features', FunctionTransformer(self._add_time_features)),
                ('derived_metrics', FunctionTransformer(self._add_derived_metrics))
            ]))
        ])
        
        return self.feature_engineering_pipeline
    
    def save_preprocessor(self, filepath: Union[str, Path]) -> None:
        """
        Save the fitted preprocessor to a file.
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self.fitted:
            raise RuntimeError("No preprocessor has been fitted.")
        
        joblib.dump(self.preprocessor, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str) -> None:
        """
        Load a preprocessor from a file.
        
        Args:
            filepath: Path to the saved preprocessor
        """
        self.preprocessor = joblib.load(filepath)
        self.fitted = True
        logger.info(f"Preprocessor loaded from {filepath}")
