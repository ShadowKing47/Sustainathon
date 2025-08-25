"""Anomaly Detection Page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Anomaly Detection | Energy Monitor",
    page_icon="⚠️"
)

def detect_anomalies_zscore(data, threshold=3.0):
    """Detect anomalies using Z-Score method."""
    mean = np.mean(data)
    std = np.std(data)
    anomalies = []
    
    for i, value in enumerate(data):
        z_score = (value - mean) / std if std != 0 else 0
        if abs(z_score) > threshold:
            anomalies.append((i, value, z_score))
    
    return anomalies

def show_anomaly_detection():
    """Display anomaly detection page."""
    st.title("⚠️ Anomaly Detection")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date",
            datetime.now() - timedelta(days=30),
            key="anomaly_start_date"
        )
    with col2:
        end_date = st.date_input(
            "End date",
            datetime.now(),
            key="anomaly_end_date"
        )
    
    # Detection settings
    st.sidebar.header("Detection Settings")
    sensitivity = st.sidebar.slider(
        "Sensitivity",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.5,
        help="Lower values detect more anomalies (more sensitive)"
    )
    
    # Equipment selection
    equipment_options = ["HVAC", "Lighting", "Machinery", "Computers", "Refrigeration"]
    selected_equipment = st.sidebar.multiselect(
        "Select Equipment to Analyze",
        equipment_options,
        default=["HVAC", "Lighting"]
    )
    
    # Generate sample data with some anomalies
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    np.random.seed(42)
    
    data = {
        'timestamp': date_range,
    }
    
    # Generate normal data with some anomalies
    for equipment in equipment_options:
        # Base pattern with daily seasonality
        base = 10 * (1 + 0.5 * np.sin(2 * np.pi * np.arange(len(date_range)) / 24))
        # Add some randomness
        noise = np.random.normal(0, 2, len(date_range))
        values = base + noise
        
        # Add some anomalies
        if equipment in selected_equipment:
            anomaly_indices = np.random.choice(len(date_range), size=len(date_range)//20, replace=False)
            values[anomaly_indices] *= np.random.uniform(2, 5, size=len(anomaly_indices))
        
        data[equipment] = values
    
    df = pd.DataFrame(data)
    
    # Detect anomalies for each selected equipment
    st.subheader("Anomaly Detection Results")
    
    for equipment in selected_equipment:
        st.write(f"### {equipment} Analysis")
        
        # Get data for this equipment
        values = df[equipment].values
        timestamps = df['timestamp']
        
        # Detect anomalies
        anomalies = detect_anomalies_zscore(values, threshold=sensitivity)
        
        # Create a plot
        fig = px.line(
            x=timestamps,
            y=values,
            title=f"{equipment} Energy Consumption with Anomalies",
            labels={"x": "Timestamp", "y": f"{equipment} (kWh)"},
            template="plotly_white"
        )
        
        # Add anomalies to the plot
        if anomalies:
            anomaly_indices = [a[0] for a in anomalies]
            anomaly_values = [a[1] for a in anomalies]
            fig.add_scatter(
                x=timestamps[anomaly_indices],
                y=anomaly_values,
                mode='markers',
                marker=dict(color='red', size=8),
                name='Anomaly'
            )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Show anomaly details in an expander
        with st.expander(f"View {equipment} Anomaly Details"):
            if anomalies:
                anomaly_data = []
                for idx, value, z_score in anomalies:
                    anomaly_data.append({
                        'Timestamp': timestamps[idx],
                        'Value (kWh)': f"{value:.2f}",
                        'Z-Score': f"{z_score:.2f}",
                        'Severity': 'High' if abs(z_score) > 4.0 else 'Medium' if abs(z_score) > 3.0 else 'Low'
                    })
                
                anomaly_df = pd.DataFrame(anomaly_data)
                st.dataframe(
                    anomaly_df,
                    column_config={
                        'Timestamp': st.column_config.DatetimeColumn(
                            "Timestamp",
                            format="YYYY-MM-DD HH:mm"
                        ),
                        'Value (kWh)': st.column_config.NumberColumn(
                            "Value (kWh)",
                            format="%.2f"
                        ),
                        'Z-Score': st.column_config.NumberColumn(
                            "Z-Score",
                            format="%.2f"
                        ),
                        'Severity': st.column_config.SelectboxColumn(
                            "Severity",
                            options=["Low", "Medium", "High"]
                        )
                    },
                    use_container_width=True
                )
                
                # Export button
                csv = anomaly_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    f"Download {equipment} Anomaly Report",
                    csv,
                    f"{equipment.lower()}_anomalies_{start_date}_to_{end_date}.csv",
                    "text/csv",
                    key=f"download_{equipment}"
                )
            else:
                st.info(f"No anomalies detected for {equipment} with current settings.")
    
    # Summary statistics
    st.sidebar.subheader("Anomaly Summary")
    st.sidebar.metric("Total Anomalies Detected", len(anomalies) if 'anomalies' in locals() else 0)
    
    if 'anomalies' in locals() and anomalies:
        avg_zscore = np.mean([a[2] for a in anomalies])
        st.sidebar.metric("Average Z-Score", f"{avg_zscore:.2f}")
        st.sidebar.metric(
            "Most Affected Equipment",
            max(selected_equipment, key=lambda x: len([a for a in anomalies if a[0] < len(df)]))
        )

if __name__ == "__main__":
    show_anomaly_detection()
