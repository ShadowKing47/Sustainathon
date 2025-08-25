""
Reusable chart components for the Energy Monitoring Dashboard.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Optional, Union

def create_time_series_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: Union[str, List[str]],
    title: str = "",
    x_title: str = "Date",
    y_title: str = "Value",
    height: int = 400
) -> go.Figure:
    """
    Create an interactive time series chart.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for x-axis (should be datetime)
        y_cols: Column name(s) for y-axis
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        height: Chart height in pixels
        
    Returns:
        plotly.graph_objects.Figure: The generated figure
    """
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    
    fig = go.Figure()
    
    for col in y_cols:
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[col],
            mode='lines',
            name=col,
            hovertemplate='%{x|%Y-%m-%d %H:%M}<br>' +
                        f'{col}: ' + '%{y:.2f} kWh<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        hovermode='x unified',
        legend_title_text='',
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig

def create_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    x_title: str = "",
    y_title: str = "",
    color: Optional[str] = None,
    barmode: str = 'group',
    height: int = 400
) -> go.Figure:
    """
    Create an interactive bar chart.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        color: Column name for color grouping
        barmode: Mode for bar display ('group', 'stack', etc.)
        height: Chart height in pixels
        
    Returns:
        plotly.graph_objects.Figure: The generated figure
    """
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color,
        title=title,
        labels={x_col: x_title, y_col: y_title},
        template='plotly_white',
        height=height
    )
    
    fig.update_layout(
        barmode=barmode,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='closest',
        legend_title_text='',
        xaxis=dict(tickangle=-45)
    )
    
    return fig

def create_pie_chart(
    df: pd.DataFrame,
    names_col: str,
    values_col: str,
    title: str = "",
    height: int = 400
) -> go.Figure:
    """
    Create an interactive pie chart.
    
    Args:
        df: DataFrame containing the data
        names_col: Column name for slice labels
        values_col: Column name for slice values
        title: Chart title
        height: Chart height in pixels
        
    Returns:
        plotly.graph_objects.Figure: The generated figure
    """
    fig = px.pie(
        df,
        names=names_col,
        values=values_col,
        title=title,
        hole=0.3,
        height=height
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label}: %{value:.2f} (%{percent})<extra></extra>'
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    
    return fig

def create_gauge_chart(
    value: float,
    min_val: float,
    max_val: float,
    title: str,
    color: str = 'blue',
    height: int = 300
) -> go.Figure:
    """
    Create an interactive gauge chart.
    
    Args:
        value: Current value
        min_val: Minimum value on the gauge
        max_val: Maximum value on the gauge
        title: Chart title
        color: Gauge color
        height: Chart height in pixels
        
    Returns:
        plotly.graph_objects.Figure: The generated figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': color},
            'steps': [
                {'range': [min_val, max_val * 0.6], 'color': 'lightgray'},
                {'range': [max_val * 0.6, max_val * 0.8], 'color': 'gray'},
                {'range': [max_val * 0.8, max_val], 'color': 'darkgray'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    title: str = "",
    x_title: str = "",
    y_title: str = "",
    colorscale: str = 'Viridis',
    height: int = 500
) -> go.Figure:
    """
    Create an interactive heatmap.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        z_col: Column name for color intensity
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        colorscale: Color scale to use
        height: Chart height in pixels
        
    Returns:
        plotly.graph_objects.Figure: The generated figure
    """
    # Pivot the data for the heatmap
    pivot_df = df.pivot(
        index=y_col,
        columns=x_col,
        values=z_col
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale=colorscale,
        hoverongaps=False,
        hovertemplate='%{x}<br>%{y}<br>%{z:.2f}<extra></extra>',
        colorbar=dict(title=z_col)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(tickangle=-45)
    )
    
    return fig
