""
Reusable filter components for the Energy Monitoring Dashboard.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import streamlit as st
import pandas as pd

def date_range_selector(
    default_days: int = 30,
    key_suffix: str = "",
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None
) -> Tuple[datetime, datetime]:
    """
    Create a date range selector widget.
    
    Args:
        default_days: Default number of days to look back
        key_suffix: Suffix for widget keys to avoid conflicts
        min_date: Minimum selectable date
        max_date: Maximum selectable date (defaults to today)
        
    Returns:
        Tuple of (start_date, end_date)
    """
    if max_date is None:
        max_date = datetime.now()
    
    default_start = max_date - timedelta(days=default_days)
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start date",
            value=default_start,
            min_value=min_date,
            max_value=max_date - timedelta(days=1),
            key=f"start_date_{key_suffix}"
        )
    
    with col2:
        end_date = st.date_input(
            "End date",
            value=max_date,
            min_value=start_date + timedelta(days=1) if start_date else min_date,
            max_value=max_date,
            key=f"end_date_{key_suffix}"
        )
    
    return start_date, end_date

def time_aggregation_selector(
    default: str = "Day",
    key_suffix: str = "",
    options: Optional[List[str]] = None
) -> str:
    """
    Create a time aggregation selector.
    
    Args:
        default: Default selected option
        key_suffix: Suffix for widget key
        options: List of aggregation options
        
    Returns:
        Selected aggregation option
    """
    if options is None:
        options = ["Hour", "Day", "Week", "Month"]
    
    return st.selectbox(
        "Time Aggregation",
        options=options,
        index=options.index(default) if default in options else 0,
        key=f"time_agg_{key_suffix}"
    )

def equipment_selector(
    equipment_list: List[str],
    default: Optional[Union[str, List[str]]] = None,
    multi: bool = True,
    key_suffix: str = ""
) -> Union[str, List[str]]:
    """
    Create an equipment selector widget.
    
    Args:
        equipment_list: List of equipment names
        default: Default selected equipment
        multi: Whether to allow multiple selections
        key_suffix: Suffix for widget key
        
    Returns:
        Selected equipment (single value or list)
    """
    if not equipment_list:
        return [] if multi else ""
    
    if multi:
        if default is None:
            default = equipment_list[:1]
        elif isinstance(default, str):
            default = [default]
        
        selected = st.multiselect(
            "Select Equipment",
            options=equipment_list,
            default=default,
            key=f"equipment_{key_suffix}"
        )
        return selected if selected else [equipment_list[0]]
    else:
        if default is None:
            default = equipment_list[0]
        
        return st.selectbox(
            "Select Equipment",
            options=equipment_list,
            index=equipment_list.index(default) if default in equipment_list else 0,
            key=f"equipment_{key_suffix}"
        )

def metric_selector(
    metrics: List[Dict[str, str]],
    default: Optional[str] = None,
    key_suffix: str = ""
) -> str:
    """
    Create a metric selector widget.
    
    Args:
        metrics: List of metric dictionaries with 'name' and 'unit' keys
        default: Default selected metric name
        key_suffix: Suffix for widget key
        
    Returns:
        Selected metric name
    """
    if not metrics:
        return ""
    
    metric_names = [m["name"] for m in metrics]
    
    if default is None:
        default = metric_names[0]
    
    selected = st.selectbox(
        "Select Metric",
        options=metric_names,
        index=metric_names.index(default) if default in metric_names else 0,
        format_func=lambda x: f"{x} ({next((m['unit'] for m in metrics if m['name'] == x), '')})",
        key=f"metric_{key_suffix}"
    )
    
    return selected

def threshold_slider(
    label: str = "Threshold",
    min_value: float = 0.0,
    max_value: float = 100.0,
    value: float = 75.0,
    step: float = 1.0,
    key_suffix: str = ""
) -> float:
    """
    Create a threshold slider widget.
    
    Args:
        label: Slider label
        min_value: Minimum value
        max_value: Maximum value
        value: Default value
        step: Step size
        key_suffix: Suffix for widget key
        
    Returns:
        Selected threshold value
    """
    return st.slider(
        label=label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        key=f"threshold_{key_suffix}"
    )

def filter_dataframe(
    df: pd.DataFrame,
    date_col: str,
    start_date: datetime,
    end_date: datetime,
    equipment_col: Optional[str] = None,
    equipment_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter a DataFrame based on date and optionally equipment.
    
    Args:
        df: Input DataFrame
        date_col: Name of the date column
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        equipment_col: Name of the equipment column (optional)
        equipment_list: List of equipment to include (optional)
        
    Returns:
        Filtered DataFrame
    """
    # Convert to datetime if not already
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Filter by date
    filtered = df[
        (df[date_col].dt.date >= start_date) & 
        (df[date_col].dt.date <= end_date)
    ].copy()
    
    # Filter by equipment if specified
    if equipment_col and equipment_list:
        filtered = filtered[filtered[equipment_col].isin(equipment_list)]
    
    return filtered

def apply_time_aggregation(
    df: pd.DataFrame,
    date_col: str,
    agg_period: str,
    numeric_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aggregate data by time period.
    
    Args:
        df: Input DataFrame
        date_col: Name of the date column
        agg_period: Aggregation period ('Hour', 'Day', 'Week', 'Month')
        numeric_cols: List of numeric columns to aggregate
        
    Returns:
        Aggregated DataFrame
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    df_agg = df.copy()
    
    # Set the date column as index
    df_agg = df_agg.set_index(date_col)
    
    # Determine the resampling frequency
    freq_map = {
        'Hour': 'H',
        'Day': 'D',
        'Week': 'W-MON',
        'Month': 'M'
    }
    
    freq = freq_map.get(agg_period, 'D')
    
    # If no numeric columns specified, use all numeric columns
    if numeric_cols is None:
        numeric_cols = df_agg.select_dtypes(include=['number']).columns.tolist()
    
    # Resample and aggregate
    if not numeric_cols:
        return df_agg.reset_index()
    
    # Group by the time period and aggregate
    df_agg = df_agg[numeric_cols].resample(freq).agg(['mean', 'sum', 'min', 'max'])
    
    # Flatten multi-index columns
    df_agg.columns = [f"{col[0]}_{col[1]}" for col in df_agg.columns]
    
    # Reset index to make date a column again
    df_agg = df_agg.reset_index()
    
    return df_agg
