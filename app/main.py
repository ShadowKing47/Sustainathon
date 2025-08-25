"""
Main Streamlit application for Energy Monitoring Dashboard
"""
import streamlit as st
from streamlit_option_menu import option_menu

# Set page config
st.set_page_config(
    page_title="Energy Monitoring Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Sidebar navigation
    with st.sidebar:
        st.title("⚡ Energy Monitor")
        
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Consumption", "Anomalies", "Forecasting", "Settings"],
            icons=["speedometer2", "graph-up", "exclamation-triangle", "graph-up-arrow", "gear"],
            default_index=0,
            styles={
                "container": {"padding": "5px"},
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "5px 0"},
            },
        )
    
    # Main content area
    st.title("Energy Monitoring Dashboard")
    
    if selected == "Dashboard":
        show_dashboard()
    elif selected == "Consumption":
        show_consumption()
    elif selected == "Anomalies":
        show_anomalies()
    elif selected == "Forecasting":
        show_forecasting()
    elif selected == "Settings":
        show_settings()

def show_dashboard():
    """Display the main dashboard view."""
    st.write("## Overview")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Power", "24.5 kW", "+2.1% vs yesterday")
    with col2:
        st.metric("Today's Usage", "356 kWh", "-5.3% vs yesterday")
    with col3:
        st.metric("Cost Today", "$42.75", "+1.2% vs yesterday")
    with col4:
        st.metric("CO₂ Emissions", "245 kg", "-3.1% vs yesterday")
    
    # Main chart area
    st.write("### Energy Consumption (Last 24 Hours)")
    # Placeholder for the chart
    st.line_chart([10, 12, 15, 18, 20, 25, 28, 30, 28, 25, 22, 20, 22, 25, 28, 30, 35, 40, 45, 40, 35, 30, 25, 20])
    
    # Bottom row
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Consumption by Equipment")
        # Placeholder for equipment pie chart
        st.bar_chart({"HVAC": 45, "Lighting": 20, "Machinery": 25, "Computers": 10})
    
    with col2:
        st.write("### Alerts")
        # Placeholder for alerts
        st.warning("⚠️ High energy consumption detected in HVAC system")
        st.info("ℹ️ Scheduled maintenance for tomorrow at 10:00 AM")

def show_consumption():
    """Display detailed consumption analysis."""
    st.write("## Energy Consumption Analysis")
    st.write("Detailed consumption metrics and visualizations will appear here.")

def show_anomalies():
    """Display anomaly detection results."""
    st.write("## Anomaly Detection")
    st.write("Detected anomalies and unusual patterns will appear here.")

def show_forecasting():
    """Display energy forecasting."""
    st.write("## Energy Forecasting")
    st.write("Energy consumption forecasts will appear here.")

def show_settings():
    """Display application settings."""
    st.write("## Settings")
    st.write("Application configuration options will appear here.")

if __name__ == "__main__":
    main()
