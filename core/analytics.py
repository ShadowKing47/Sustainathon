"""
Analytics and statistical functions for the Energy Monitoring System.
Provides comprehensive energy consumption analysis, pattern detection, and forecasting.
"""
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, OPTICS, SpectralClustering
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz
import holidays
from scipy.signal import periodogram, welch
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Import config
from .config_loader import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnergyAnalytics:
    """
    Comprehensive energy data analysis and statistical functions.
    
    This class provides methods for time series analysis, anomaly detection,
    pattern recognition, and energy consumption forecasting.
    """
    
    def __init__(self, data: pd.DataFrame = None, config: dict = None):
        """
        Initialize the EnergyAnalytics class.
        
        Args:
            data: Optional DataFrame with time series data
            config: Configuration dictionary with analysis parameters
        """
        self.data = data
        self.config = config or {}
        self._results = {}
        self._models = {}
        self._visualizations = {}
        self._metrics = {}
        
        # Set default configuration
        self._set_default_config()
    
    def _set_default_config(self) -> None:
        """Set default configuration parameters."""
        defaults = {
            'time_column': 'timestamp',
            'value_column': 'consumption',
            'outlier_threshold': 3.0,
            'anomaly_contamination': 0.05,
            'forecast_horizon': 24,
            'forecast_freq': 'H',
            'train_test_split': 0.8,
            'cross_validation_folds': 5,
            'clustering': {
                'n_clusters': 3,
                'max_clusters': 10,
                'random_state': 42
            },
            'seasonality': {
                'daily': True,
                'weekly': True,
                'monthly': True,
                'yearly': False
            },
            'visualization': {
                'width': 1200,
                'height': 600,
                'template': 'plotly_white',
                'colors': {
                    'actual': '#1f77b4',
                    'forecast': '#ff7f0e',
                    'anomaly': '#d62728',
                    'trend': '#2ca02c',
                    'seasonal': '#9467bd',
                    'residual': '#8c564b'
                }
            }
        }
        
        # Update with user-provided config
        for key, value in self.config.items():
            if isinstance(value, dict) and key in defaults and isinstance(defaults[key], dict):
                defaults[key].update(value)
            else:
                defaults[key] = value
        
        self.config = defaults
    
    def set_data(self, data: pd.DataFrame, time_column: str = None, value_column: str = None) -> None:
        """
        Set the data for analysis with optional column specification.
        
        Args:
            data: DataFrame containing the time series data
            time_column: Name of the datetime column
            value_column: Name of the value column to analyze
        """
        self.data = data.copy()
        
        # Update column names in config if provided
        if time_column:
            self.config['time_column'] = time_column
        if value_column:
            self.config['value_column'] = value_column
        
        # Ensure proper datetime index
        time_col = self.config['time_column']
        if time_col in self.data.columns and not pd.api.types.is_datetime64_any_dtype(self.data[time_col]):
            self.data[time_col] = pd.to_datetime(self.data[time_col])
        
        # Set datetime index if not already set
        if time_col in self.data.columns and self.data.index.name != time_col:
            self.data = self.data.set_index(time_col)
        
        # Sort by index
        self.data = self.data.sort_index()
        
        # Ensure numeric values
        value_col = self.config['value_column']
        if value_col in self.data.columns and not pd.api.types.is_numeric_dtype(self.data[value_col]):
            self.data[value_col] = pd.to_numeric(self.data[value_col], errors='coerce')
        
        logger.info(f"Data set with {len(self.data)} records from {self.data.index.min()} to {self.data.index.max()}")
    
    def get_summary_statistics(self, resample: str = None) -> Dict:
        """
        Calculate comprehensive summary statistics for the data.
        
        Args:
            resample: Optional resampling frequency (e.g., 'D' for daily, 'H' for hourly)
            
        Returns:
            Dictionary containing summary statistics and time-based aggregations
        """
        if self.data is None:
            raise ValueError("No data available for analysis")
            
        value_col = self.config['value_column']
        data = self.data[value_col] if value_col in self.data.columns else self.data.iloc[:, 0]
        
        # Basic statistics
        stats = {
            'count': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            '25%': data.quantile(0.25),
            'median': data.median(),
            '75%': data.quantile(0.75),
            'max': data.max(),
            'sum': data.sum(),
            'range': data.max() - data.min(),
            'cv': data.std() / data.mean() if data.mean() != 0 else np.nan,  # Coefficient of variation
            'skewness': data.skew(),
            'kurtosis': data.kurtosis(),
            'zeros': (data == 0).sum(),
            'missing': data.isna().sum(),
            'missing_pct': data.isna().mean() * 100
        }
        
        # Time-based aggregations if datetime index
        if isinstance(data.index, pd.DatetimeIndex):
            # Resample if requested
            if resample:
                resampled = data.resample(resample).sum()
                stats.update({
                    f'mean_{resample}': resampled.mean(),
                    f'std_{resample}': resampled.std(),
                    f'min_{resample}': resampled.min(),
                    f'max_{resample}': resampled.max(),
                    f'median_{resample}': resampled.median(),
                })
            
            # Time-based metrics
            stats.update({
                'start_date': data.index.min(),
                'end_date': data.index.max(),
                'duration_days': (data.index.max() - data.index.min()).days,
                'records_per_day': len(data) / ((data.index.max() - data.index.min()).days + 1) if len(data) > 1 else 1,
            })
            
            # Peak analysis
            stats.update(self._calculate_peak_metrics(data))
        
        return stats
    
    def _calculate_peak_metrics(self, series: pd.Series) -> Dict:
        """
        Calculate peak-related metrics for energy consumption data.
        
        Args:
            series: Time series data with datetime index
            
        Returns:
            Dictionary containing peak metrics including load factor, peak frequency, etc.
        """
        if len(series) < 2:
            return {}
            
        # Basic peak metrics
        peak_value = series.max()
        peak_time = series.idxmax() if not series.isna().all() else pd.NaT
        
        # Rolling peak detection with adaptive window size
        rolling_window = min(24, max(4, len(series) // 100))  # Adaptive window size (4-24)
        rolling_peaks = series.rolling(window=rolling_window, center=True, min_periods=1).max()
        
        # Calculate load duration curve
        sorted_values = np.sort(series.dropna())[::-1]  # Sort descending
        load_duration = pd.Series(sorted_values, index=np.linspace(0, 100, len(sorted_values)))
        
        # Calculate peak demand metrics
        daily_max = series.resample('D').max()
        monthly_max = series.resample('M').max()
        
        return {
            'peak_value': float(peak_value) if not pd.isna(peak_value) else None,
            'peak_time': peak_time,
            'mean_peak_value': float(rolling_peaks.mean()) if len(rolling_peaks) > 0 else None,
            'peak_frequency': float((series > (0.8 * peak_value)).mean()) if peak_value > 0 else 0,
            'load_factor': float(series.mean() / peak_value) if peak_value != 0 else 0,
            'peak_to_average_ratio': float(peak_value / series.mean()) if series.mean() != 0 else 0,
            'daily_peak_avg': float(daily_max.mean()) if len(daily_max) > 0 else None,
            'monthly_peak_avg': float(monthly_max.mean()) if len(monthly_max) > 0 else None,
            'load_duration_curve': load_duration.to_dict(),
            'peak_hours': self._identify_peak_hours(series),
            'base_load': float(series.quantile(0.1)) if len(series) > 0 else None,
            'peak_load_ratio': float(series.quantile(0.9) / series.quantile(0.1)) if len(series) > 1 and series.quantile(0.1) > 0 else None
        }
        
    def _identify_peak_hours(self, series: pd.Series, threshold: float = 0.9) -> Dict:
        """Identify peak hours based on historical data."""
        if not isinstance(series.index, pd.DatetimeIndex):
            return {}
            
        # Extract hour of day for each timestamp
        hour_series = series.copy()
        hour_series.index = hour_series.index.hour
        
        # Calculate hourly statistics
        hourly_stats = hour_series.groupby(level=0).agg(['mean', 'std', 'count'])
        if len(hourly_stats) == 0:
            return {}
            
        # Identify peak hours (hours where mean is above threshold percentile)
        threshold_value = hourly_stats['mean'].quantile(threshold)
        peak_hours = hourly_stats[hourly_stats['mean'] >= threshold_value].index.tolist()
        
        return {
            'peak_hours': sorted(peak_hours),
            'off_peak_hours': [h for h in range(24) if h not in peak_hours],
            'peak_hour_load': float(hourly_stats['mean'].max()) if len(hourly_stats) > 0 else None,
            'off_peak_load': float(hourly_stats['mean'].min()) if len(hourly_stats) > 0 else None
        }
    
    def analyze_time_series(self, resample: str = 'H', decompose: bool = True) -> Dict:
        """
        Perform comprehensive time series analysis.
        
        Args:
            resample: Resampling frequency (e.g., 'H' for hourly, 'D' for daily)
            decompose: Whether to perform time series decomposition
            
        Returns:
            Dictionary with analysis results including stationarity tests, seasonality, etc.
        """
        if self.data is None:
            raise ValueError("No data available for analysis")
            
        value_col = self.config['value_column']
        series = self.data[value_col] if value_col in self.data.columns else self.data.iloc[:, 0]
        
        # Resample if needed
        if resample:
            series = series.resample(resample).mean()
            
        # Handle missing values
        if series.isna().any():
            series = series.interpolate(method='time')
            
        results = {
            'summary_statistics': self.get_summary_statistics(resample=resample),
            'stationarity_tests': self._test_stationarity(series),
            'autocorrelation': self._calculate_autocorrelation(series),
            'seasonality': self._analyze_seasonality(series)
        }
        
        # Time series decomposition
        if decompose and len(series) > 24:  # Need sufficient data points
            try:
                decomposition = self.decompose_time_series(series)
                results['decomposition'] = {
                    'trend': decomposition['trend'].describe().to_dict(),
                    'seasonal': decomposition['seasonal'].describe().to_dict(),
                    'residual': decomposition['resid'].describe().to_dict()
                }
            except Exception as e:
                logger.warning(f"Time series decomposition failed: {str(e)}")
                
        return results
        
    def _test_stationarity(self, series: pd.Series, max_lag: int = None) -> Dict:
        """Perform stationarity tests on the time series."""
        if len(series) < 2:
            return {}
            
        if max_lag is None:
            max_lag = min(12, len(series) // 5)  # Default to 12 lags or 20% of data length
            
        results = {}
        
        # Augmented Dickey-Fuller test
        try:
            adf_result = adfuller(series.dropna(), maxlag=max_lag)
            results['adf'] = {
                'test_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': {f'{k}': v for k, v in adf_result[4].items()},
                'is_stationary': adf_result[1] < 0.05
            }
        except Exception as e:
            logger.warning(f"ADF test failed: {str(e)}")
            
        # KPSS test
        try:
            kpss_result = kpss(series.dropna(), nlags=max_lag)
            results['kpss'] = {
                'test_statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': {f'{k}': v for k, v in kpss_result[3].items()},
                'is_stationary': kpss_result[1] > 0.05
            }
        except Exception as e:
            logger.warning(f"KPSS test failed: {str(e)}")
            
        return results
        
    def _calculate_autocorrelation(self, series: pd.Series, max_lags: int = 50) -> Dict:
        """Calculate autocorrelation and partial autocorrelation."""
        if len(series) < 2:
            return {}
            
        max_lags = min(max_lags, len(series) // 2 - 1)
        
        # Calculate ACF and PACF
        acf_vals, acf_conf = acf(series.dropna(), nlags=max_lags, alpha=0.05)
        pacf_vals, pacf_conf = pacf(series.dropna(), nlags=max_lags, alpha=0.05)
        
        # Find significant lags (outside 95% confidence interval)
        significant_lags = {
            'acf': [i for i, val in enumerate(acf_vals) if abs(val) > acf_conf[i][1] - val],
            'pacf': [i for i, val in enumerate(pacf_vals) if abs(val) > pacf_conf[i][1] - val]
        }
        
        return {
            'acf': acf_vals.tolist(),
            'pacf': pacf_vals.tolist(),
            'acf_confidence': acf_conf.tolist(),
            'pacf_confidence': pacf_conf.tolist(),
            'significant_lags': significant_lags,
            'suggested_arima_order': self._suggest_arima_order(series)
        }
        
    def _suggest_arima_order(self, series: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Dict:
        """Suggest ARIMA order parameters based on ACF/PACF analysis."""
        try:
            from pmdarima import auto_arima
            
            model = auto_arima(
                series,
                start_p=1, start_q=1,
                max_p=max_p, max_q=max_q, max_d=max_d,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            
            return {
                'p': model.order[0],
                'd': model.order[1],
                'q': model.order[2],
                'aic': model.aic(),
                'bic': model.bic()
            }
        except Exception as e:
            logger.warning(f"ARIMA order suggestion failed: {str(e)}")
            return {}
    
    def _analyze_seasonality(self, series: pd.Series) -> Dict:
        """Analyze seasonality in the time series."""
        if len(series) < 24:  # Minimum 24 data points for seasonality analysis
            return {}
            
        # Detect seasonality periods
        periods = self._detect_seasonal_periods(series)
        
        # Calculate seasonal strength
        try:
            result = seasonal_decompose(series.dropna(), period=periods.get('daily', 24) if 'daily' in periods else 24)
            seasonal_strength = max(0, 1 - (result.resid.var() / (result.trend + result.seasonal).var()))
        except:
            seasonal_strength = None
            
        return {
            'detected_periods': periods,
            'seasonal_strength': seasonal_strength,
            'is_seasonal': seasonal_strength is not None and seasonal_strength > 0.3
        }
        
    def _detect_seasonal_periods(self, series: pd.Series) -> Dict:
        """Detect potential seasonal periods in the time series."""
        periods = {}
        
        # Check for daily seasonality (24 hours)
        if len(series) >= 24 * 7:  # At least one week of hourly data
            periods['daily'] = 24
            
        # Check for weekly seasonality (168 hours)
        if len(series) >= 24 * 7 * 4:  # At least 4 weeks of data
            periods['weekly'] = 24 * 7
            
        # Check for monthly patterns (approximately 30 days)
        if len(series) >= 24 * 30 * 3:  # At least 3 months of data
            periods['monthly'] = 24 * 30
            
        # Use FFT to detect other potential periods
        try:
            fft_vals = np.abs(fft(series.fillna(series.mean()).values))
            freqs = fftfreq(len(series))
            
            # Find peaks in the frequency domain
            peaks = np.where(fft_vals > np.percentile(fft_vals, 95))[0]
            for peak in peaks:
                if 0 < peak < len(series) / 2:  # Ignore negative frequencies and DC component
                    period = int(round(1 / abs(freqs[peak])))
                    if 2 <= period <= len(series) // 2:  # Reasonable period range
                        periods[f'period_{period}'] = period
        except:
            pass
            
        return periods
        
    def analyze_correlations(self, target_column: str = None, method: str = 'pearson', 
                           max_lags: int = 24, threshold: float = 0.3) -> Dict:
        """
        Analyze correlations between time series variables.
        
        Args:
            target_column: Target column for correlation analysis
            method: Correlation method ('pearson', 'spearman', 'kendall')
            max_lags: Maximum number of lags to consider for cross-correlation
            threshold: Minimum absolute correlation value to report
            
        Returns:
            Dictionary with correlation analysis results
        """
        if self.data is None:
            raise ValueError("No data available for analysis")
            
        if target_column is None:
            target_column = self.config['value_column']
            
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {}
            
        results = {
            'target': target_column,
            'method': method,
            'correlation_matrix': {},
            'lagged_correlations': {},
            'top_correlations': []
        }
        
        # Calculate correlation matrix
        corr_matrix = self.data[numeric_cols].corr(method=method)
        results['correlation_matrix'] = corr_matrix.to_dict()
        
        # Find top correlations with target
        target_corrs = corr_matrix[target_column].drop(target_column, errors='ignore')
        target_corrs = target_corrs[abs(target_corrs) >= threshold].sort_values(
            key=abs, ascending=False
        )
        
        for col, corr in target_corrs.items():
            results['top_correlations'].append({
                'feature': col,
                'correlation': corr,
                'lag': 0,
                'is_significant': True
            })
        
        # Calculate lagged correlations
        if max_lags > 0 and len(self.data) > max_lags * 2:
            for col in numeric_cols:
                if col == target_column:
                    continue
                    
                # Calculate cross-correlation
                xcorr = sm.tsa.stattools.ccf(
                    self.data[target_column].fillna(method='ffill').fillna(0),
                    self.data[col].fillna(method='ffill').fillna(0),
                    adjusted=False
                )
                
                # Find significant lags
                significant_lags = np.where(abs(xcorr) >= threshold)[0]
                for lag in significant_lags:
                    if 0 < lag <= max_lags:  # Only consider positive lags up to max_lags
                        results['top_correlations'].append({
                            'feature': col,
                            'correlation': float(xcorr[lag]),
                            'lag': int(lag),
                            'is_significant': True
                        })
        
        # Sort by absolute correlation
        results['top_correlations'].sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return results

    def check_stationarity(self, series: pd.Series, test_type: str = 'adf', 
                          max_lags: int = None, 
                          regression: str = 'c') -> Dict[str, float]:
        """
        Check stationarity of a time series.
        
        Args:
            series: Time series data
            test_type: Type of test ('adf' for Augmented Dickey-Fuller, 'kpss' for KPSS)
            max_lags: Maximum number of lags to use (None for automatic selection)
            regression: Type of regression ('c' for constant, 'ct' for constant and trend, 'ctt' for constant, linear and quadratic trend)
            
        Returns:
            Dictionary with test results
        """
        if test_type.lower() == 'adf':
            result = adfuller(series, maxlag=max_lags, regression=regression)
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'n_lags': result[2],
                'n_obs': result[3],
                'critical_values': result[4],
                'stationary': result[1] <= 0.05  # Null hypothesis: unit root exists (non-stationary)
            }
        
        elif test_type.lower() == 'kpss':
            result = kpss(series, regression=regression, nlags='auto' if max_lags is None else max_lags)
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'n_lags': result[2],
                'critical_values': result[3],
                'stationary': result[1] > 0.05  # Null hypothesis: stationary
            }
        
        else:
            raise ValueError("Invalid test_type. Use 'adf' or 'kpss'.")
    
    def decompose_time_series(self, series: pd.Series, 
                            freq: int = None, 
                            model: str = 'additive') -> Dict[str, pd.Series]:
        """
        Decompose a time series into trend, seasonal, and residual components.
        
        Args:
            series: Time series data with datetime index
            freq: Seasonal period (if None, will attempt to infer)
            model: 'additive' or 'multiplicative' decomposition
            
        Returns:
            Dictionary with decomposed components
        """
        if freq is None:
            # Try to infer frequency
            if hasattr(series.index, 'freq') and series.index.freq is not None:
                freq = series.index.freq
            else:
                # Default to 24 for hourly data, 7 for daily, 12 for monthly
                if len(series) > 24 * 7:  # More than a week of hourly data
                    freq = 24
                elif len(series) > 30 * 3:  # More than 3 months of daily data
                    freq = 7
                elif len(series) > 12 * 2:  # More than 2 years of monthly data
                    freq = 12
                else:
                    freq = 1  # Default to 1 if can't determine
        
        try:
            decomposition = seasonal_decompose(
                series.dropna(),
                period=freq,
                model=model,
                extrapolate_trend=True
            )
            
            return {
                'observed': decomposition.observed,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'model': model
            }
            
        except Exception as e:
            logger.error(f"Error in time series decomposition: {e}")
            return {}
    
    def calculate_correlations(self, method: str = 'pearson', 
                             target_col: str = None, 
                             top_n: int = 10) -> pd.DataFrame:
        """
        Calculate correlations between numeric columns.
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            target_col: Optional target column to calculate correlations with
            top_n: Number of top correlations to return (if target_col is specified)
            
        Returns:
            Correlation matrix or top N correlations with target
        """
        if self.data is None:
            raise ValueError("No data available. Set data using set_data() first.")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return pd.DataFrame()
        
        corr_matrix = self.data[numeric_cols].corr(method=method)
        
        if target_col and target_col in corr_matrix.columns:
            # Return top N correlations with target
            corr_with_target = corr_matrix[target_col].drop(target_col, errors='ignore')
            corr_with_target = corr_with_target.sort_values(key=abs, ascending=False)
            return corr_with_target.head(top_n)
        
        return corr_matrix
    
    def detect_seasonality(self, series: pd.Series, 
                          max_lag: int = 100, 
                          threshold: float = 0.5) -> Dict:
        """
        Detect seasonality in a time series using autocorrelation.
        
        Args:
            series: Time series data
            max_lag: Maximum lag to check for seasonality
            threshold: Threshold for significant autocorrelation
            
        Returns:
            Dictionary with seasonality detection results
        """
        if len(series) < 2 * max_lag:
            max_lag = len(series) // 2
        
        # Calculate autocorrelation
        acf_values = acf(series.dropna(), nlags=max_lag, fft=True)
        
        # Find significant peaks (above threshold)
        significant_lags = np.where(np.abs(acf_values) > threshold)[0]
        significant_lags = significant_lags[significant_lags > 0]  # Exclude lag 0
        
        # If no significant lags found, return empty result
        if len(significant_lags) == 0:
            return {
                'is_seasonal': False,
                'seasonal_periods': [],
                'strength': 0.0
            }
        
        # Find the most significant seasonal periods
        seasonal_periods = []
        for lag in significant_lags:
            # Check if this lag is a multiple of a smaller lag (to avoid duplicates)
            is_multiple = any(lag % p == 0 for p in seasonal_periods)
            if not is_multiple:
                seasonal_periods.append(lag)
        
        # Calculate overall seasonality strength
        strength = np.mean(np.abs(acf_values[significant_lags]))
        
        return {
            'is_seasonal': True,
            'seasonal_periods': sorted(seasonal_periods),
            'strength': strength,
            'acf': acf_values
        }
    
    def calculate_energy_metrics(self, power_col: str, 
                               time_col: str = None, 
                               interval_minutes: float = 15) -> pd.DataFrame:
        """
        Calculate energy metrics from power data.
        
        Args:
            power_col: Column name with power data (in kW)
            time_col: Column with timestamps (if None, use index)
            interval_minutes: Time interval in minutes (default: 15)
            
        Returns:
            DataFrame with calculated metrics
        """
        if self.data is None:
            raise ValueError("No data available. Set data using set_data() first.")
        
        df = self.data.copy()
        
        if time_col is not None and time_col in df.columns:
            df = df.set_index(time_col)
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be a DatetimeIndex or provide a valid time_col")
        
        # Ensure data is sorted by time
        df = df.sort_index()
        
        # Calculate energy in kWh (power * time)
        # Time delta in hours
        if len(df) > 1:
            time_delta_hours = (df.index.to_series().diff().dt.total_seconds().fillna(interval_minutes * 60)) / 3600
        else:
            time_delta_hours = pd.Series(interval_minutes / 60, index=df.index)
        
        # Calculate energy for each interval (kWh)
        df['energy_kwh'] = df[power_col] * time_delta_hours
        
        # Calculate cumulative energy
        df['cumulative_energy_kwh'] = df['energy_kwh'].cumsum()
        
        # Calculate daily energy
        daily_energy = df['energy_kwh'].resample('D').sum()
        
        # Calculate peak demand (rolling window of 15/30/60 minutes)
        for window in [15, 30, 60]:
            df[f'peak_demand_{window}min'] = (
                df[power_col].rolling(f'{window}T', min_periods=1).max()
            )
        
        # Calculate load factor (average load / peak load)
        if 'peak_demand_60min' in df.columns:
            df['load_factor'] = df[power_col] / df['peak_demand_60min'].replace(0, np.nan)
        
        return df
    
    def detect_anomalies(self, data: pd.DataFrame = None, 
                        columns: List[str] = None, 
                        method: str = 'zscore', 
                        **kwargs) -> pd.DataFrame:
        """
        Detect anomalies in the data.
        
        Args:
            data: Input DataFrame (if None, use self.data)
            columns: Columns to analyze (if None, use all numeric columns)
            method: Method for anomaly detection ('zscore', 'iqr', 'isolation_forest')
            **kwargs: Additional arguments for the detection method
            
        Returns:
            DataFrame with anomaly scores and flags
        """
        df = data if data is not None else self.data
        
        if df is None:
            raise ValueError("No data available. Provide data or use set_data() first.")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = pd.DataFrame(index=df.index)
        
        if method == 'zscore':
            threshold = kwargs.get('threshold', 3.0)
            window = kwargs.get('window', 24)
            
            for col in columns:
                # Calculate rolling statistics
                rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                rolling_std = df[col].rolling(window=window, min_periods=1).std()
                
                # Calculate z-scores
                z_scores = (df[col] - rolling_mean) / rolling_std
                
                # Detect anomalies
                results[f'{col}_anomaly_score'] = z_scores.abs()
                results[f'{col}_is_anomaly'] = results[f'{col}_anomaly_score'] > threshold
        
        elif method == 'iqr':
            threshold = kwargs.get('threshold', 1.5)
            
            for col in columns:
                # Calculate quartiles and IQR
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                
                # Define bounds
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                # Calculate anomaly scores (distance from bounds in IQR units)
                scores = pd.Series(0, index=df.index)
                below = df[col] < lower_bound
                above = df[col] > upper_bound
                scores[below] = (lower_bound - df[col][below]) / iqr
                scores[above] = (df[col][above] - upper_bound) / iqr
                
                # Store results
                results[f'{col}_anomaly_score'] = scores
                results[f'{col}_is_anomaly'] = (below | above)
        
        elif method == 'isolation_forest':
            contamination = kwargs.get('contamination', 0.01)
            n_estimators = kwargs.get('n_estimators', 100)
            
            # Prepare data
            X = df[columns].copy()
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit isolation forest
            clf = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42
            )
            
            # Predict anomalies
            preds = clf.fit_predict(X_scaled)
            scores = -clf.score_samples(X_scaled)  # Lower is more anomalous
            
            # Store results
            results['anomaly_score'] = scores
            results['is_anomaly'] = preds == -1
            
            # Also store individual feature contributions to anomalies
            if hasattr(clf, 'feature_importances_'):
                for i, col in enumerate(columns):
                    results[f'{col}_importance'] = clf.feature_importances_[i]
        
        else:
            raise ValueError(f"Unsupported anomaly detection method: {method}")
        
        return results
    
    def cluster_consumption_patterns(self, data: pd.DataFrame = None, 
                                    columns: List[str] = None,
                                    n_clusters: int = 3,
                                    method: str = 'kmeans',
                                    **kwargs) -> Dict:
        """
        Cluster consumption patterns in the data.
        
        Args:
            data: Input DataFrame (if None, use self.data)
            columns: Columns to use for clustering (if None, use all numeric columns)
            n_clusters: Number of clusters to find
            method: Clustering method ('kmeans', 'dbscan')
            **kwargs: Additional arguments for the clustering method
            
        Returns:
            Dictionary with clustering results
        """
        df = data if data is not None else self.data
        
        if df is None:
            raise ValueError("No data available. Provide data or use set_data() first.")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Prepare data
        X = df[columns].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reduce dimensionality with PCA (optional)
        use_pca = kwargs.get('use_pca', False)
        pca_components = kwargs.get('pca_components', 0.95)
        
        if use_pca and len(columns) > 2:
            pca = PCA(n_components=pca_components, random_state=42)
            X_reduced = pca.fit_transform(X_scaled)
            explained_variance = sum(pca.explained_variance_ratio_)
            logger.info(f"Reduced to {X_reduced.shape[1]} components, explaining {explained_variance:.1%} of variance")
        else:
            X_reduced = X_scaled
        
        # Apply clustering
        if method == 'kmeans':
            n_init = kwargs.get('n_init', 10)
            random_state = kwargs.get('random_state', 42)
            
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=n_init,
                random_state=random_state
            )
            
            cluster_labels = kmeans.fit_predict(X_reduced)
            cluster_centers = kmeans.cluster_centers_
            
            # Calculate inertia (within-cluster sum of squares)
            inertia = kmeans.inertia_
            
            # Calculate silhouette score (if possible)
            from sklearn.metrics import silhouette_score
            if len(set(cluster_labels)) > 1:  # At least 2 clusters needed
                silhouette_avg = silhouette_score(X_reduced, cluster_labels)
            else:
                silhouette_avg = None
            
            # Calculate cluster sizes
            cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
            
            return {
                'labels': cluster_labels,
                'centers': cluster_centers,
                'inertia': inertia,
                'silhouette_score': silhouette_avg,
                'cluster_sizes': cluster_sizes.to_dict(),
                'method': 'kmeans',
                'n_clusters': n_clusters,
                'features': columns,
                'pca_used': use_pca,
                'pca_components': pca_components if use_pca else None,
                'explained_variance': explained_variance if use_pca else None
            }
            
        elif method == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(X_reduced)
            
            # Calculate number of clusters (ignoring noise if present)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            # Calculate cluster sizes
            cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
            
            # Calculate silhouette score (if possible)
            if n_clusters > 1:  # At least 2 non-noise clusters needed
                from sklearn.metrics import silhouette_score
                # Only use points that are not noise for silhouette score
                mask = cluster_labels != -1
                if sum(mask) > 1:
                    try:
                        silhouette_avg = silhouette_score(X_reduced[mask], cluster_labels[mask])
                    except:
                        silhouette_avg = None
                else:
                    silhouette_avg = None
            else:
                silhouette_avg = None
            
            return {
                'labels': cluster_labels,
                'n_clusters': n_clusters,
                'n_noise': int((cluster_labels == -1).sum()),
                'silhouette_score': silhouette_avg,
                'cluster_sizes': cluster_sizes.to_dict(),
                'method': 'dbscan',
                'features': columns,
                'pca_used': use_pca,
                'pca_components': pca_components if use_pca else None,
                'explained_variance': explained_variance if use_pca else None
            }
            
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
    
    def calculate_energy_savings(self, baseline_consumption: float,
                               current_consumption: float,
                               baseline_period_days: float = 30,
                               current_period_days: float = 30) -> Dict[str, float]:
        """
        Calculate energy savings between two periods.
        
        Args:
            baseline_consumption: Baseline energy consumption (kWh)
            current_consumption: Current energy consumption (kWh)
            baseline_period_days: Duration of baseline period in days
            current_period_days: Duration of current period in days
            
        Returns:
            Dictionary with energy savings metrics
        """
        # Normalize to daily consumption
        baseline_daily = baseline_consumption / baseline_period_days
        current_daily = current_consumption / current_period_days
        
        # Calculate savings
        absolute_savings = baseline_consumption - (current_consumption * (baseline_period_days / current_period_days))
        relative_savings = (absolute_savings / baseline_consumption) * 100 if baseline_consumption > 0 else 0
        
        return {
            'baseline_consumption': baseline_consumption,
            'current_consumption': current_consumption,
            'baseline_daily': baseline_daily,
            'current_daily': current_daily,
            'absolute_savings': absolute_savings,
            'relative_savings': relative_savings,
            'baseline_period_days': baseline_period_days,
            'current_period_days': current_period_days
        }
    
    def forecast_energy_consumption(self, series: pd.Series, 
                                   forecast_horizon: int = 24,
                                   freq: str = 'H',
                                   model_type: str = 'prophet',
                                   train_test_split: float = 0.8) -> Dict:
        """
        Forecast energy consumption using a time series model.
        
        Args:
            series: Time series data with datetime index
            forecast_horizon: Number of periods to forecast
            freq: Frequency of the time series ('H' for hourly, 'D' for daily, etc.)
            model_type: Type of model to use ('prophet', 'arima', 'ets')
            train_test_split: Fraction of data to use for training
            
        Returns:
            Dictionary with forecast results
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series must have a DatetimeIndex")
        
        # Ensure series is sorted
        series = series.sort_index()
        
        # Split into train and test sets
        train_size = int(len(series) * train_test_split)
        train, test = series.iloc[:train_size], series.iloc[train_size:]
        
        if model_type.lower() == 'prophet':
            try:
                from prophet import Prophet
                
                # Prepare data for Prophet
                train_df = train.reset_index()
                train_df.columns = ['ds', 'y']
                
                # Fit model
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    seasonality_mode='multiplicative'
                )
                
                model.fit(train_df)
                
                # Create future dataframe for forecasting
                future = model.make_future_dataframe(periods=forecast_horizon, freq=freq)
                
                # Make forecast
                forecast = model.predict(future)
                
                # Extract predictions
                forecast_series = forecast.set_index('ds')['yhat']
                
                # Calculate prediction intervals
                lower_bound = forecast.set_index('ds')['yhat_lower']
                upper_bound = forecast.set_index('ds')['yhat_upper']
                
                # Calculate metrics on test set if available
                metrics = {}
                if len(test) > 0:
                    # Get predictions for test period
                    test_forecast = forecast_series[test.index]
                    
                    # Calculate metrics
                    metrics = {
                        'mae': mean_absolute_error(test, test_forecast[:len(test)]),
                        'rmse': np.sqrt(mean_squared_error(test, test_forecast[:len(test)])),
                        'mape': mean_absolute_percentage_error(test, test_forecast[:len(test)]),
                        'r2': r2_score(test, test_forecast[:len(test)])
                    }
                
                return {
                    'model': model,
                    'forecast': forecast_series,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'train': train,
                    'test': test,
                    'metrics': metrics,
                    'model_type': 'prophet'
                }
                
            except ImportError:
                logger.warning("Prophet not installed. Install with: pip install prophet")
                return {}
        
        elif model_type.lower() == 'arima':
            try:
                from pmdarima import auto_arima
                
                # Fit auto_arima
                model = auto_arima(
                    train,
                    seasonal=True,
                    m=24 if freq == 'H' else 7 if freq == 'D' else 12,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore'
                )
                
                # Make forecast
                forecast, conf_int = model.predict(
                    n_periods=forecast_horizon,
                    return_conf_int=True
                )
                
                # Create index for forecast
                last_date = series.index[-1]
                if freq == 'H':
                    freq_pd = 'H'
                elif freq == 'D':
                    freq_pd = 'D'
                elif freq == 'M':
                    freq_pd = 'M'
                else:
                    freq_pd = freq
                
                forecast_index = pd.date_range(
                    start=last_date + pd.Timedelta(1, unit=freq_pd),
                    periods=forecast_horizon,
                    freq=freq_pd
                )
                
                forecast_series = pd.Series(forecast, index=forecast_index)
                lower_bound = pd.Series(conf_int[:, 0], index=forecast_index)
                upper_bound = pd.Series(conf_int[:, 1], index=forecast_index)
                
                # Calculate metrics on test set if available
                metrics = {}
                if len(test) > 0:
                    # Get predictions for test period
                    test_forecast = []
                    history = train.copy()
                    
                    for t in range(len(test)):
                        model = auto_arima(
                            history,
                            seasonal=True,
                            m=24 if freq == 'H' else 7 if freq == 'D' else 12,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore'
                        )
                        
                        # One-step forecast
                        fc = model.predict(n_periods=1)
                        test_forecast.append(fc[0])
                        
                        # Update history with actual value
                        history = history.append(pd.Series([test.iloc[t]], index=[test.index[t]]))
                    
                    test_forecast = pd.Series(test_forecast, index=test.index[:len(test_forecast)])
                    
                    # Calculate metrics
                    metrics = {
                        'mae': mean_absolute_error(test[:len(test_forecast)], test_forecast),
                        'rmse': np.sqrt(mean_squared_error(test[:len(test_forecast)], test_forecast)),
                        'mape': mean_absolute_percentage_error(test[:len(test_forecast)], test_forecast),
                        'r2': r2_score(test[:len(test_forecast)], test_forecast)
                    }
                
                return {
                    'model': model,
                    'forecast': forecast_series,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'train': train,
                    'test': test,
                    'metrics': metrics,
                    'model_type': 'arima'
                }
                
            except ImportError:
                logger.warning("pmdarima not installed. Install with: pip install pmdarima")
                return {}
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
