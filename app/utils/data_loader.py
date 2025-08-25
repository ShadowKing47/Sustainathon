""
Data loading and preprocessing utilities for the Energy Monitoring Dashboard.
"""
import os
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import logging
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and preprocessing of energy monitoring data."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data'
        )
        self.raw_data_dir = os.path.join(self.data_dir, 'raw')
        self.processed_data_dir = os.path.join(self.data_dir, 'processed')
        self._create_directories()
        
        # Data schema mapping
        self.schema = self._load_schema()
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def _load_schema(self) -> Dict:
        """Load data schema from YAML file."""
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'config',
            'schema.yaml'
        )
        
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file with automatic encoding detection.
        
        Args:
            filename: Name of the CSV file (with or without .csv extension)
            **kwargs: Additional arguments to pass to pandas.read_csv()
            
        Returns:
            Loaded DataFrame
        """
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        filepath = os.path.join(self.raw_data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding, **kwargs)
                logger.info(f"Successfully loaded {filename} with {encoding} encoding")
                return self._preprocess_data(df, filename)
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Failed to load {filename} with any of the supported encodings")
    
    def _preprocess_data(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Preprocess the loaded data based on filename.
        
        Args:
            df: Input DataFrame
            filename: Name of the source file
            
        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert column names to lowercase with underscores
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Apply schema transformations if available
        file_key = os.path.splitext(os.path.basename(filename))[0]
        
        if file_key in self.schema:
            schema = self.schema[file_key]
            
            # Convert date/time columns
            for col, col_type in schema.get('dtypes', {}).items():
                if col in df.columns:
                    if col_type == 'datetime':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif col_type == 'numeric':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle common date/time columns
        datetime_columns = [col for col in df.columns if 'date' in col or 'time' in col or 'timestamp' in col]
        for col in datetime_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Sort by date if available
        if datetime_columns:
            df = df.sort_values(by=datetime_columns[0])
        
        return df
    
    def load_energy_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all energy-related data files.
        
        Returns:
            Dictionary of DataFrames with keys as dataset names
        """
        data_files = {
            'consumption': 'energy_consumption_2023.csv',
            'cost_billing': 'Cost_n_Billing_data.csv',
            'environmental': 'Updated_Environmental & External Factors.csv',
            'iot': 'IOT_and_sensor_Data.csv',
            'equipment': 'Equipment_n_Appliance_data.csv',
            'updates': 'Updates_Cost_n_billing.csv'
        }
        
        data = {}
        
        for name, filename in data_files.items():
            try:
                data[name] = self.load_csv(filename)
                logger.info(f"Successfully loaded {filename}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
                data[name] = pd.DataFrame()
        
        return data
    
    def save_processed_data(self, df: pd.DataFrame, name: str) -> None:
        """
        Save processed data to the processed data directory.
        
        Args:
            df: DataFrame to save
            name: Name of the dataset (without extension)
        """
        if not name.endswith('.parquet'):
            name += '.parquet'
        
        filepath = os.path.join(self.processed_data_dir, name)
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")
    
    def load_processed_data(self, name: str) -> pd.DataFrame:
        """
        Load processed data from the processed data directory.
        
        Args:
            name: Name of the dataset (without extension)
            
        Returns:
            Loaded DataFrame
        """
        if not name.endswith('.parquet'):
            name += '.parquet'
        
        filepath = os.path.join(self.processed_data_dir, name)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Processed data not found: {filepath}")
        
        return pd.read_parquet(filepath)
    
    def get_equipment_list(self) -> List[str]:
        """
        Get a list of unique equipment from the data.
        
        Returns:
            List of equipment names
        """
        try:
            iot_data = self.load_csv('IOT_and_sensor_Data.csv')
            if 'equipment' in iot_data.columns:
                return sorted(iot_data['equipment'].unique().tolist())
            
            equipment_data = self.load_csv('Equipment_n_Appliance_data.csv')
            equipment_columns = [col.split('_')[0] for col in equipment_data.columns if '_status' in col.lower()]
            return sorted(equipment_columns)
        except Exception as e:
            logger.warning(f"Could not load equipment list: {str(e)}")
            return ['HVAC', 'Lighting', 'Machinery', 'Computers', 'Refrigeration']

def resample_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_cols: Union[str, List[str]],
    freq: str = 'D',
    agg_func: str = 'mean'
) -> pd.DataFrame:
    """
    Resample time series data to a different frequency.
    
    Args:
        df: Input DataFrame
        date_col: Name of the datetime column
        value_cols: Column(s) to aggregate
        freq: Pandas frequency string (e.g., 'H', 'D', 'W', 'M')
        agg_func: Aggregation function ('sum', 'mean', 'max', 'min')
        
    Returns:
        Resampled DataFrame
    """
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    
    # Make sure date_col is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Set datetime as index
    df = df.set_index(date_col).sort_index()
    
    # Resample
    if agg_func == 'sum':
        resampled = df[value_cols].resample(freq).sum()
    elif agg_func == 'mean':
        resampled = df[value_cols].resample(freq).mean()
    elif agg_func == 'max':
        resampled = df[value_cols].resample(freq).max()
    elif agg_func == 'min':
        resampled = df[value_cols].resample(freq).min()
    else:
        raise ValueError(f"Unsupported aggregation function: {agg_func}")
    
    return resampled.reset_index()

def detect_anomalies(
    df: pd.DataFrame,
    value_col: str,
    method: str = 'zscore',
    threshold: float = 3.0,
    window: int = 24
) -> pd.Series:
    """
    Detect anomalies in a time series.
    
    Args:
        df: Input DataFrame
        value_col: Name of the value column
        method: Detection method ('zscore' or 'iqr')
        threshold: Threshold for anomaly detection
        window: Window size for rolling calculations
        
    Returns:
        Boolean Series indicating anomalies
    """
    if method == 'zscore':
        # Calculate rolling statistics
        rolling_mean = df[value_col].rolling(window=window, min_periods=1).mean()
        rolling_std = df[value_col].rolling(window=window, min_periods=1).std()
        
        # Calculate z-scores
        z_scores = (df[value_col] - rolling_mean) / rolling_std
        
        # Detect anomalies
        return (z_scores.abs() > threshold).astype(int)
    
    elif method == 'iqr':
        # Calculate quartiles and IQR
        q1 = df[value_col].quantile(0.25)
        q3 = df[value_col].quantile(0.75)
        iqr = q3 - q1
        
        # Define bounds
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        # Detect anomalies
        return ((df[value_col] < lower_bound) | (df[value_col] > upper_bound)).astype(int)
    
    else:
        raise ValueError(f"Unsupported anomaly detection method: {method}")
