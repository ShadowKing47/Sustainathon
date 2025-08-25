"""
Energy Consumption Analysis Module

This module provides functions for analyzing and visualizing energy consumption patterns,
including time-based aggregations, correlation analysis, and KPI calculations.
"""
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, time
import pytz

class ConsumptionAnalyzer:
    """Class for analyzing energy consumption patterns and metrics."""
    
    def __init__(self, data: pd.DataFrame, timestamp_col: str = 'timestamp', 
                 consumption_col: str = 'consumption', temperature_col: str = 'temperature',
                 occupancy_col: str = 'occupancy'):
        """
        Initialize the ConsumptionAnalyzer.
        
        Args:
            data: DataFrame containing the energy consumption data
            timestamp_col: Name of the timestamp column
            consumption_col: Name of the energy consumption column
            temperature_col: Name of the temperature column
            occupancy_col: Name of the occupancy column
        """
        self.data = data.copy()
        self.timestamp_col = timestamp_col
        self.consumption_col = consumption_col
        self.temperature_col = temperature_col
        self.occupancy_col = occupancy_col
        
        # Ensure timestamp is in datetime format
        if self.timestamp_col in self.data.columns:
            self.data[self.timestamp_col] = pd.to_datetime(self.data[self.timestamp_col])
            self.data = self.data.set_index(self.timestamp_col).sort_index()
    
    def get_consumption_summary(self, freq: str = 'D') -> pd.DataFrame:
        """
        Calculate consumption summaries for the specified time frequency.
        
        Args:
            freq: Time frequency for aggregation ('D' for daily, 'W' for weekly, 'M' for monthly)
            
        Returns:
            DataFrame with consumption statistics
        """
        if self.consumption_col not in self.data.columns:
            raise ValueError(f"Consumption column '{self.consumption_col}' not found in data")
            
        resampled = self.data[self.consumption_col].resample(freq)
        
        summary = pd.DataFrame({
            'total_consumption': resampled.sum(),
            'avg_consumption': resampled.mean(),
            'max_consumption': resampled.max(),
            'min_consumption': resampled.min(),
            'std_consumption': resampled.std(),
            'consumption_95th': resampled.quantile(0.95),
            'consumption_5th': resampled.quantile(0.05)
        })
        
        return summary
    
    def analyze_peak_usage(self, peak_hours: tuple = (8, 20), 
                          threshold: float = 0.8) -> Dict:
        """
        Analyze peak vs off-peak usage patterns.
        
        Args:
            peak_hours: Tuple of (start_hour, end_hour) for peak period
            threshold: Threshold for peak identification (fraction of max consumption)
            
        Returns:
            Dictionary with peak analysis results
        """
        if self.consumption_col not in self.data.columns:
            raise ValueError(f"Consumption column '{self.consumption_col}' not found in data")
            
        # Calculate hourly consumption
        hourly = self.data[self.consumption_col].resample('H').mean()
        
        # Identify peak and off-peak periods
        peak_mask = (hourly.index.hour >= peak_hours[0]) & (hourly.index.hour < peak_hours[1])
        peak_consumption = hourly[peak_mask]
        offpeak_consumption = hourly[~peak_mask]
        
        # Calculate peak threshold
        peak_threshold = threshold * hourly.max()
        
        # Calculate metrics
        result = {
            'total_consumption': {
                'peak': peak_consumption.sum(),
                'off_peak': offpeak_consumption.sum(),
                'peak_ratio': peak_consumption.sum() / hourly.sum()
            },
            'average_consumption': {
                'peak': peak_consumption.mean(),
                'off_peak': offpeak_consumption.mean(),
                'ratio': peak_consumption.mean() / offpeak_consumption.mean() if offpeak_consumption.mean() > 0 else float('inf')
            },
            'peak_periods': {
                'threshold': peak_threshold,
                'peak_hours': f"{peak_hours[0]}:00-{peak_hours[1]}:00",
                'peak_detection_threshold': threshold
            },
            'peak_load_factor': peak_consumption.mean() / peak_consumption.max() if peak_consumption.max() > 0 else 0
        }
        
        return result
    
    def analyze_correlations(self, window: int = 24) -> Dict:
        """
        Analyze correlations between consumption, temperature, and occupancy.
        
        Args:
            window: Window size for rolling correlations
            
        Returns:
            Dictionary with correlation analysis results
        """
        required_cols = [self.consumption_col]
        if self.temperature_col in self.data.columns:
            required_cols.append(self.temperature_col)
        if self.occupancy_col in self.data.columns:
            required_cols.append(self.occupancy_col)
            
        if len(required_cols) < 2:
            raise ValueError("At least one of temperature or occupancy data is required for correlation analysis")
            
        # Resample to hourly data if needed
        data = self.data[required_cols].resample('H').mean()
        
        # Calculate correlations
        correlations = data.corr(method='pearson')
        
        # Calculate rolling correlations if window is specified
        rolling_corrs = {}
        if len(required_cols) > 1 and window > 1:
            for col in required_cols:
                if col != self.consumption_col:
                    rolling_corrs[f"rolling_corr_{col}"] = data[self.consumption_col].rolling(window=window).corr(
                        data[col])
        
        return {
            'pearson_correlations': correlations.to_dict(),
            'rolling_correlations': {k: v.dropna().to_dict() for k, v in rolling_corrs.items()},
            'analysis_window_hours': window
        }
    
    def calculate_kpis(self) -> Dict:
        """
        Calculate key performance indicators for energy consumption.
        
        Returns:
            Dictionary with KPIs
        """
        if self.consumption_col not in self.data.columns:
            raise ValueError(f"Consumption column '{self.consumption_col}' not found in data")
            
        daily = self.data[self.consumption_col].resample('D').sum()
        
        # Basic metrics
        kpis = {
            'total_consumption': daily.sum(),
            'avg_daily_consumption': daily.mean(),
            'max_daily_consumption': daily.max(),
            'min_daily_consumption': daily.min(),
            'cv_daily_consumption': daily.std() / daily.mean() if daily.mean() > 0 else 0,
            'peak_to_average_ratio': daily.max() / daily.mean() if daily.mean() > 0 else 0
        }
        
        # Add temperature-normalized metrics if available
        if self.temperature_col in self.data.columns:
            # Calculate heating and cooling degree days
            base_temp = 18  # Base temperature in Celsius
            self.data['hdd'] = np.maximum(base_temp - self.data[self.temperature_col], 0)
            self.data['cdd'] = np.maximum(self.data[self.temperature_col] - base_temp, 0)
            
            # Calculate energy intensity metrics
            kpis.update({
                'consumption_per_hdd': daily.sum() / self.data['hdd'].sum() if self.data['hdd'].sum() > 0 else 0,
                'consumption_per_cdd': daily.sum() / self.data['cdd'].sum() if self.data['cdd'].sum() > 0 else 0
            })
        
        # Add occupancy-based metrics if available
        if self.occupancy_col in self.data.columns:
            total_occupancy = self.data[self.occupancy_col].sum()
            kpis['consumption_per_occupant'] = (
                daily.sum() / total_occupancy if total_occupancy > 0 else 0
            )
        
        return kpis
    
    def plot_consumption_profile(self, freq: str = 'D', figsize: tuple = (12, 6)) -> plt.Figure:
        """
        Plot energy consumption profile.
        
        Args:
            freq: Time frequency for plotting ('D' for daily, 'W' for weekly, 'M' for monthly)
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        if self.consumption_col not in self.data.columns:
            raise ValueError(f"Consumption column '{self.consumption_col}' not found in data")
            
        plt.figure(figsize=figsize)
        
        if freq == 'H':
            # Hourly profile (average by hour of day)
            hourly = self.data[self.consumption_col].groupby(self.data.index.hour).mean()
            plt.plot(hourly.index, hourly.values, marker='o')
            plt.xlabel('Hour of Day')
            plt.title('Average Hourly Energy Consumption Profile')
        elif freq == 'D':
            # Daily profile (average by day of week)
            daily = self.data[self.consumption_col].groupby(self.data.index.dayofweek).mean()
            plt.plot(daily.index, daily.values, marker='o')
            plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            plt.title('Average Daily Energy Consumption Profile')
        elif freq == 'W':
            # Weekly profile (time series)
            weekly = self.data[self.consumption_col].resample('W').sum()
            weekly.plot()
            plt.title('Weekly Energy Consumption')
        elif freq == 'M':
            # Monthly profile
            monthly = self.data[self.consumption_col].resample('M').sum()
            monthly.plot(kind='bar')
            plt.title('Monthly Energy Consumption')
            plt.xticks(rotation=45)
        
        plt.ylabel('Energy Consumption (kWh)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_heatmap(self, freq: str = 'daily', figsize: tuple = (12, 8)) -> plt.Figure:
        """
        Create a heatmap of energy consumption patterns.
        
        Args:
            freq: Frequency for heatmap ('daily' or 'weekly')
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        if self.consumption_col not in self.data.columns:
            raise ValueError(f"Consumption column '{self.consumption_col}' not found in data")
            
        plt.figure(figsize=figsize)
        
        if freq == 'daily':
            # Daily heatmap (hour of day vs day of week)
            pivot = self.data.pivot_table(
                index=self.data.index.hour,
                columns=self.data.index.dayofweek,
                values=self.consumption_col,
                aggfunc='mean'
            )
            sns.heatmap(pivot, cmap='YlOrRd', 
                       yticklabels=range(24),
                       xticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            plt.title('Daily Consumption Heatmap (Hour of Day vs Day of Week)')
            plt.ylabel('Hour of Day')
            plt.xlabel('Day of Week')
            
        elif freq == 'weekly':
            # Weekly heatmap (day of week vs week of year)
            pivot = self.data.groupby([
                self.data.index.isocalendar().week,
                self.data.index.dayofweek
            ])[self.consumption_col].mean().unstack()
            
            sns.heatmap(pivot, cmap='YlOrRd',
                       yticklabels=pivot.index,
                       xticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            plt.title('Weekly Consumption Heatmap (Week of Year vs Day of Week)')
            plt.ylabel('Week of Year')
            plt.xlabel('Day of Week')
        
        plt.tight_layout()
        return plt.gcf()


def calculate_energy_savings(baseline: pd.Series, comparison: pd.Series, 
                            baseline_label: str = 'Baseline',
                            comparison_label: str = 'Comparison') -> Dict:
    """
    Calculate energy savings between two periods.
    
    Args:
        baseline: Baseline energy consumption series
        comparison: Comparison energy consumption series
        baseline_label: Label for baseline period
        comparison_label: Label for comparison period
        
    Returns:
        Dictionary with energy savings metrics
    """
    if len(baseline) == 0 or len(comparison) == 0:
        raise ValueError("Input series cannot be empty")
        
    baseline_mean = baseline.mean()
    comparison_mean = comparison.mean()
    
    if baseline_mean == 0:
        raise ValueError("Baseline mean consumption cannot be zero")
    
    absolute_savings = baseline_mean - comparison_mean
    percent_savings = (absolute_savings / baseline_mean) * 100
    
    return {
        'baseline': {
            'label': baseline_label,
            'mean_consumption': baseline_mean,
            'total_consumption': baseline.sum()
        },
        'comparison': {
            'label': comparison_label,
            'mean_consumption': comparison_mean,
            'total_consumption': comparison.sum()
        },
        'savings': {
            'absolute': absolute_savings,
            'percent': percent_savings,
            'is_improvement': percent_savings > 0
        },
        'confidence_interval': {
            't_stat': stats.ttest_ind(baseline, comparison, equal_var=False).statistic,
            'p_value': stats.ttest_ind(baseline, comparison, equal_var=False).pvalue
        }
    }
