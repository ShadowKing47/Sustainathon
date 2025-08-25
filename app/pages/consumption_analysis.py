"""Consumption Analysis Page."""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Consumption Analysis | Energy Monitor",
    page_icon="📊"
)

def show_consumption_analysis():
    """Display consumption analysis page."""
    st.title("📊 Consumption Analysis")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date",
            datetime.now() - timedelta(days=30),
            key="start_date"
        )
    with col2:
        end_date = st.date_input(
            "End date",
            datetime.now(),
            key="end_date"
        )
    
    # Time aggregation
    time_agg = st.selectbox(
        "Time Aggregation",
        ["Hourly", "Daily", "Weekly", "Monthly"],
        index=1
    )
    
    # Equipment filter
    equipment_options = ["All"] + ["HVAC", "Lighting", "Machinery", "Computers", "Refrigeration"]
    selected_equipment = st.multiselect(
        "Select Equipment",
        equipment_options,
        default=["All"]
    )
    
    # Generate sample data (replace with actual data loading)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    sample_data = []
    
    for date in date_range:
        for hour in range(24):
            timestamp = date + timedelta(hours=hour)
            sample_data.append({
                "timestamp": timestamp,
                "HVAC": 10 + 5 * (hour / 24) * (1 + 0.1 * (timestamp.day % 7)),
                "Lighting": 2 + 3 * (hour / 24) * (1 + 0.05 * (timestamp.day % 7)),
                "Machinery": 8 + 10 * (timestamp.weekday() < 5) * (8 <= hour < 18),
                "Computers": 3 + 2 * (timestamp.weekday() < 5) * (8 <= hour < 18),
                "Refrigeration": 1.5 + 0.5 * (timestamp.month % 2)
            })
    
    df = pd.DataFrame(sample_data)
    
    # Filter data based on selection
    if "All" not in selected_equipment:
        df = df[["timestamp"] + selected_equipment]
    
    # Resample based on time aggregation
    freq_map = {
        "Hourly": "H",
        "Daily": "D",
        "Weekly": "W-MON",
        "Monthly": "M"
    }
    
    df_resampled = df.set_index("timestamp").resample(freq_map[time_agg]).sum()
    
    # Plot energy consumption
    st.subheader(f"Energy Consumption ({time_agg})")
    fig = px.area(
        df_resampled,
        x=df_resampled.index,
        y=df_resampled.columns,
        title=f"Energy Consumption by Equipment ({time_agg} View)",
        labels={"value": "Energy (kWh)", "timestamp": "Date"},
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Energy breakdown
    st.subheader("Energy Breakdown")
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of energy distribution
        total_energy = df_resampled.sum().sum()
        energy_by_equipment = df_resampled.sum().sort_values(ascending=False)
        
        fig_pie = px.pie(
            values=energy_by_equipment,
            names=energy_by_equipment.index,
            title="Energy Distribution by Equipment",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Stats
        st.metric("Total Energy Consumed", f"{total_energy:,.0f} kWh")
        st.metric("Average Daily Consumption", f"{total_energy/len(df_resampled):.1f} kWh/{time_agg.lower()}")
        
        # Top consumer
        top_consumer = energy_by_equipment.idxmax()
        top_percent = (energy_by_equipment.max() / total_energy) * 100
        st.metric(
            "Top Energy Consumer",
            f"{top_consumer}",
            f"{top_percent:.1f}% of total"
        )
    
    # Data table
    st.subheader("Consumption Data")
    st.dataframe(df_resampled.style.format("{:.2f}"), use_container_width=True)

if __name__ == "__main__":
    show_consumption_analysis()
