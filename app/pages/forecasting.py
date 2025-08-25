"""Energy Forecasting Page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet

# Page config
st.set_page_config(
    page_title="Energy Forecasting | Energy Monitor",
    page_icon="📈"
)

def generate_forecast(data, periods=24, freq='H'):
    """Generate forecast using Prophet."""
    try:
        # Prepare data for Prophet
        df = data.rename(columns={'timestamp': 'ds', 'value': 'y'})
        
        # Initialize and fit model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative'
        )
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq)
        
        # Generate forecast
        forecast = model.predict(future)
        
        return model, forecast
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        return None, None

def show_forecasting():
    """Display forecasting page."""
    st.title("📈 Energy Forecasting")
    
    # Sidebar settings
    st.sidebar.header("Forecast Settings")
    
    # Forecast horizon
    forecast_horizon = st.sidebar.selectbox(
        "Forecast Horizon",
        ["24 hours", "7 days", "30 days", "90 days"],
        index=0
    )
    
    # Map horizon to periods and frequency
    horizon_map = {
        "24 hours": (24, 'H'),
        "7 days": (7, 'D'),
        "30 days": (30, 'D'),
        "90 days": (90, 'D')
    }
    periods, freq = horizon_map[forecast_horizon]
    
    # Confidence interval
    uncertainty = st.sidebar.slider(
        "Confidence Interval",
        min_value=0.7,
        max_value=0.99,
        value=0.9,
        step=0.05,
        help="Width of the uncertainty interval"
    )
    
    # Equipment selection
    equipment_options = ["HVAC", "Lighting", "Machinery", "Computers", "Refrigeration"]
    selected_equipment = st.sidebar.multiselect(
        "Select Equipment to Forecast",
        equipment_options,
        default=["HVAC", "Lighting"]
    )
    
    # Generate sample time series data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    np.random.seed(42)
    
    # Create sample data with trend and seasonality
    def generate_equipment_data(base, trend, seasonal_amp, noise_scale, date_range):
        t = np.linspace(0, 10, len(date_range))
        # Base + trend + seasonality + noise
        values = (
            base + 
            trend * t + 
            seasonal_amp * np.sin(2 * np.pi * t / 24) +  # Daily seasonality
            seasonal_amp * 0.5 * np.sin(2 * np.pi * t / (24*7)) +  # Weekly seasonality
            np.random.normal(0, noise_scale, len(date_range))  # Noise
        )
        return np.maximum(values, 0)  # Ensure no negative values
    
    # Equipment parameters
    equipment_params = {
        "HVAC": {"base": 10, "trend": 0.01, "seasonal_amp": 5, "noise_scale": 1},
        "Lighting": {"base": 5, "trend": 0.005, "seasonal_amp": 2, "noise_scale": 0.5},
        "Machinery": {"base": 8, "trend": 0.02, "seasonal_amp": 3, "noise_scale": 1.5},
        "Computers": {"base": 3, "trend": 0.003, "seasonal_amp": 1, "noise_scale": 0.3},
        "Refrigeration": {"base": 2, "trend": 0.001, "seasonal_amp": 0.5, "noise_scale": 0.2}
    }
    
    # Generate data for each equipment
    data = {'timestamp': date_range}
    for eq in equipment_options:
        params = equipment_params[eq]
        data[eq] = generate_equipment_data(
            params["base"],
            params["trend"],
            params["seasonal_amp"],
            params["noise_scale"],
            date_range
        )
    
    df = pd.DataFrame(data)
    
    # Display forecast for each selected equipment
    for equipment in selected_equipment:
        st.subheader(f"{equipment} Forecast")
        
        # Prepare data for Prophet
        prophet_df = df[['timestamp', equipment]].rename(columns={equipment: 'value'})
        
        # Generate forecast
        model, forecast = generate_forecast(prophet_df, periods=periods, freq=freq[0])
        
        if model and forecast is not None:
            # Create plot
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=prophet_df['timestamp'],
                y=prophet_df['value'],
                mode='lines',
                name='Historical',
                line=dict(color='#1f77b4')
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='#ff7f0e', dash='dash')
            ))
            
            # Uncertainty interval
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name=f'Confidence Interval ({int(uncertainty*100)}%)'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{equipment} Energy Consumption Forecast",
                xaxis_title="Date",
                yaxis_title="Energy Consumption (kWh)",
                template="plotly_white",
                hovermode="x unified",
                showlegend=True
            )
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Show forecast metrics
            st.subheader("Forecast Summary")
            
            # Calculate metrics
            last_actual = prophet_df['value'].iloc[-1]
            forecast_avg = forecast['yhat'].mean()
            forecast_max = forecast['yhat'].max()
            forecast_min = forecast['yhat'].min()
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Consumption", f"{last_actual:.1f} kWh")
            col2.metric("Forecast Average", f"{forecast_avg:.1f} kWh", 
                       f"{(forecast_avg - last_actual)/last_actual*100:.1f}%")
            col3.metric("Forecast Range", 
                       f"{forecast_min:.1f} - {forecast_max:.1f} kWh")
            
            # Show forecast table
            with st.expander("View Forecast Data"):
                forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                forecast_display.columns = ['Timestamp', 'Forecast', 'Lower Bound', 'Upper Bound']
                forecast_display['Timestamp'] = forecast_display['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(
                    forecast_display,
                    column_config={
                        'Timestamp': 'Timestamp',
                        'Forecast': st.column_config.NumberColumn(
                            'Forecast (kWh)',
                            format="%.2f"
                        ),
                        'Lower Bound': st.column_config.NumberColumn(
                            'Lower Bound (kWh)',
                            format="%.2f"
                        ),
                        'Upper Bound': st.column_config.NumberColumn(
                            'Upper Bound (kWh)',
                            format="%.2f"
                        )
                    },
                    use_container_width=True
                )
                
                # Download button
                csv = forecast_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    f"Download {equipment} Forecast",
                    csv,
                    f"{equipment.lower()}_forecast_{forecast_horizon.replace(' ', '_')}.csv",
                    "text/csv",
                    key=f"download_forecast_{equipment}"
                )
    
    # Show data quality note
    st.sidebar.info(
        "Note: This is a demonstration with synthetic data. "
        "For production use, replace with your actual energy consumption data."
    )

if __name__ == "__main__":
    show_forecasting()
