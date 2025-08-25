""
Visualization utilities for the Energy Monitoring Dashboard.
"""
from typing import Dict, List, Optional, Union, Tuple
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

# Color palette for consistent styling
COLOR_PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'tertiary': '#2ca02c',
    'quaternary': '#d62728',
    'background': '#f9f9f9',
    'grid': '#e1e1e1',
    'text': '#333333',
    'anomaly': '#ff0000'
}

# Template for consistent plot styling
PLOT_TEMPLATE = {
    'layout': {
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {'color': COLOR_PALETTE['text']},
        'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50},
        'xaxis': {
            'showgrid': True,
            'gridcolor': COLOR_PALETTE['grid'],
            'zeroline': False,
            'showline': True,
            'linecolor': COLOR_PALETTE['grid'],
            'mirror': True
        },
        'yaxis': {
            'showgrid': True,
            'gridcolor': COLOR_PALETTE['grid'],
            'zeroline': False,
            'showline': True,
            'linecolor': COLOR_PALETTE['grid'],
            'mirror': True
        },
        'hoverlabel': {
            'bgcolor': 'white',
            'font_size': 12,
            'font_family': 'Arial'
        },
        'legend': {
            'orientation': 'h',
            'y': 1.1,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'bottom'
        }
    }
}

class EnergyVisualizer:
    """Class for creating energy-related visualizations."""
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize the visualizer with a theme.
        
        Args:
            theme: Plotly theme to use
        """
        self.theme = theme
    
    def create_time_series(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_cols: Union[str, List[str]],
        title: str = "",
        x_title: str = "Date",
        y_title: str = "Value",
        height: int = 400,
        show_legend: bool = True,
        template: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create an interactive time series plot.
        
        Args:
            df: DataFrame containing the data
            x_col: Column name for x-axis
            y_cols: Column name(s) for y-axis
            title: Plot title
            x_title: X-axis title
            y_title: Y-axis title
            height: Plot height in pixels
            show_legend: Whether to show the legend
            template: Custom template to override defaults
            
        Returns:
            Plotly Figure object
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
                hovertemplate=f'%{{x|%Y-%m-%d %H:%M}}<br>{col}: %{{y:.2f}}<extra></extra>'
            ))
        
        # Apply template or use default
        fig.update_layout(
            template=template or PLOT_TEMPLATE,
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            height=height,
            showlegend=show_legend,
            hovermode='x unified',
            xaxis={
                'rangeselector': {
                    'buttons': [
                        {'count': 1, 'label': '1d', 'step': 'day', 'stepmode': 'backward'},
                        {'count': 7, 'label': '1w', 'step': 'day', 'stepmode': 'backward'},
                        {'count': 1, 'label': '1m', 'step': 'month', 'stepmode': 'backward'},
                        {'count': 6, 'label': '6m', 'step': 'month', 'stepmode': 'backward'},
                        {'step': 'all'}
                    ]
                },
                'rangeslider': {'visible': True},
                'type': 'date'
            }
        )
        
        return fig
    
    def create_bar_chart(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str = "",
        x_title: str = "",
        y_title: str = "",
        color: Optional[str] = None,
        barmode: str = 'group',
        height: int = 400,
        template: Optional[Dict] = None
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
            template: Custom template to override defaults
            
        Returns:
            Plotly Figure object
        """
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            color=color,
            title=title,
            labels={x_col: x_title, y_col: y_title},
            height=height,
            template=template or PLOT_TEMPLATE,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig.update_layout(
            barmode=barmode,
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode='closest',
            xaxis=dict(tickangle=-45)
        )
        
        return fig
    
    def create_pie_chart(
        self,
        df: pd.DataFrame,
        names_col: str,
        values_col: str,
        title: str = "",
        height: int = 400,
        hole: float = 0.3,
        template: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create an interactive pie chart.
        
        Args:
            df: DataFrame containing the data
            names_col: Column name for slice labels
            values_col: Column name for slice values
            title: Chart title
            height: Chart height in pixels
            hole: Size of the hole in the middle (0-1)
            template: Custom template to override defaults
            
        Returns:
            Plotly Figure object
        """
        fig = px.pie(
            df,
            names=names_col,
            values=values_col,
            title=title,
            hole=hole,
            height=height,
            template=template or PLOT_TEMPLATE,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='%{label}: %{value:.2f} (%{percent})<extra></extra>'
        )
        
        return fig
    
    def create_heatmap(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        title: str = "",
        x_title: str = "",
        y_title: str = "",
        colorscale: str = 'Viridis',
        height: int = 500,
        template: Optional[Dict] = None
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
            template: Custom template to override defaults
            
        Returns:
            Plotly Figure object
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
            xaxis=dict(tickangle=-45),
            template=template or PLOT_TEMPLATE
        )
        
        return fig
    
    def create_gauge_chart(
        self,
        value: float,
        min_val: float,
        max_val: float,
        title: str,
        color: str = 'blue',
        height: int = 300,
        template: Optional[Dict] = None
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
            template: Custom template to override defaults
            
        Returns:
            Plotly Figure object
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
            margin=dict(l=20, r=20, t=50, b=20),
            template=template or PLOT_TEMPLATE
        )
        
        return fig
    
    def create_anomaly_chart(
        self,
        df: pd.DataFrame,
        time_col: str,
        value_col: str,
        anomaly_col: str,
        title: str = "Anomaly Detection",
        x_title: str = "Date",
        y_title: str = "Value",
        height: int = 400,
        template: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create a time series chart with anomalies highlighted.
        
        Args:
            df: DataFrame containing the data
            time_col: Column name for time values
            value_col: Column name for the main values
            anomaly_col: Column name indicating anomalies (1 for anomaly, 0 otherwise)
            title: Chart title
            x_title: X-axis title
            y_title: Y-axis title
            height: Chart height in pixels
            template: Custom template to override defaults
            
        Returns:
            Plotly Figure object with anomalies highlighted
        """
        # Create figure with secondary y-axis for anomalies
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add main time series
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[value_col],
                name=value_col,
                mode='lines',
                line=dict(color=COLOR_PALETTE['primary'])
            ),
            secondary_y=False
        )
        
        # Add anomalies if any
        if anomaly_col in df.columns and df[anomaly_col].sum() > 0:
            anomalies = df[df[anomaly_col] == 1]
            
            fig.add_trace(
                go.Scatter(
                    x=anomalies[time_col],
                    y=anomalies[value_col],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(
                        color=COLOR_PALETTE['anomaly'],
                        size=8,
                        line=dict(width=1, color='white')
                    )
                ),
                secondary_y=False
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            height=height,
            showlegend=True,
            hovermode='x unified',
            template=template or PLOT_TEMPLATE,
            xaxis={
                'rangeselector': {
                    'buttons': [
                        {'count': 1, 'label': '1d', 'step': 'day', 'stepmode': 'backward'},
                        {'count': 7, 'label': '1w', 'step': 'day', 'stepmode': 'backward'},
                        {'count': 1, 'label': '1m', 'step': 'month', 'stepmode': 'backward'},
                        {'step': 'all'}
                    ]
                },
                'rangeslider': {'visible': True},
                'type': 'date'
            }
        )
        
        return fig

def create_dashboard_metrics(
    metrics: List[Dict[str, any]],
    columns: int = 4
) -> None:
    """
    Display a row of metric cards in the dashboard.
    
    Args:
        metrics: List of metric dictionaries with 'label', 'value', and optional 'delta'
        columns: Number of columns in the row (1-4)
    """
    if not metrics:
        return
    
    cols = st.columns(min(columns, 4))
    
    for i, metric in enumerate(metrics):
        with cols[i % len(cols)]:
            st.metric(
                label=metric.get('label', ''),
                value=metric.get('value', 'N/A'),
                delta=metric.get('delta', None)
            )

def create_card(title: str, content, width: int = 1) -> None:
    """
    Create a card component in the dashboard.
    
    Args:
        title: Card title
        content: Content to display in the card (text, plot, etc.)
        width: Width of the card (1-4, where 4 is full width)
    """
    with st.container():
        st.markdown(f"### {title}")
        st.markdown("---")
        st.write(content)
        st.markdown("")
        
        # Add some styling
        st.markdown(
            """
            <style>
            .stContainer {
                border-radius: 10px;
                padding: 20px;
                background-color: white;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
