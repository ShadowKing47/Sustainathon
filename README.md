# Energy Monitoring System

A comprehensive energy monitoring solution with real-time dashboard and alerting capabilities.

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
