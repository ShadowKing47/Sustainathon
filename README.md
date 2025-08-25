# Energy Consumption Analytics & Alert System

## Overview
This project provides an end-to-end framework for analyzing, forecasting, and monitoring building energy consumption. By leveraging advanced analytics and machine learning, it enables better decision-making for energy efficiency, cost reduction, and sustainability. The system integrates an interactive Streamlit dashboard for visualization and a Telegram bot for real-time anomaly alerts, ensuring both accessibility and timely responses.

## Importance
Energy management is critical in today's world due to increasing energy costs, environmental concerns, and the push for sustainable practices. Traditional monitoring methods often lack actionable insights, making it difficult to identify inefficiencies or respond to unusual consumption patterns.

This project addresses these challenges by:
- Providing insights into energy usage trends and their correlation with environmental and operational factors such as temperature and occupancy.
- Detecting anomalies in consumption that may signal equipment failures, inefficiencies, or abnormal operational behavior.
- Forecasting demand, allowing better planning for peak hours and optimized use of energy resources.
- Enabling automation through alerts, reducing the reliance on manual monitoring and ensuring proactive management.

## Value Proposition
- **For Facility Managers**: Improved visibility into building performance, enabling quick identification of inefficiencies.
- **For Businesses**: Reduced operational costs through optimized consumption and tariff-aware planning.
- **For Sustainability Goals**: Data-driven decisions that help reduce carbon footprint and align with green building practices.
- **For Researchers & Developers**: A modular, extensible codebase for experimenting with anomaly detection, forecasting models, and IoT data streams.

## Features
- **Data Preprocessing**: Cleans and transforms raw energy datasets for reliable analysis.
- **Consumption Analytics**: Statistical summaries and visualizations of energy usage patterns.
- **Anomaly Detection**: Identification of unusual consumption behaviors in real-time.
- **Forecasting Models**: Predictive insights for short-term and long-term energy usage.
- **Streamlit Dashboard**: Interactive interface for data exploration and decision support.
- **Telegram Alerts**: Instant notifications for anomalies or threshold breaches.

## Project Structure

```
energy_monitoring_project/
├── app/                          # Streamlit Dashboard
├── bot/                          # Telegram Bot
├── core/                         # Business logic
├── config/                       # Configuration files
├── data/                         # Data storage
├── models/                       # ML models
├── tests/                        # Test suite
├── notebooks/                    # Jupyter notebooks
└── scripts/                      # Utility scripts
```

## Future Scope
- Integration with IoT sensors for real-time data streaming.
- Adaptive forecasting models based on seasonal and occupancy variations.
- Multi-building or campus-wide energy optimization.
- Advanced tariff optimization strategies for dynamic pricing models.

## Setup

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables in `.env`
5. Run the application:
   ```bash
   # Start Streamlit dashboard
   streamlit run app/main.py
   
   # Start Telegram bot
   python bot/main.py
   ```

## Configuration

Copy `.env.example` to `.env` and update the values:

```
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Data Paths
DATA_PATH=./data/raw/
PROCESSED_PATH=./data/processed/

# Model Paths
MODEL_PATH=./models/
```

## Development

- Run tests: `pytest`
- Format code: `black .`
- Check linting: `flake8`
