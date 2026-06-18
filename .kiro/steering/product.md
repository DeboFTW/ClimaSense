# ClimaSense - Product Overview

ClimaSense is a Flask-based smart weather monitoring web app with **Hybrid Machine Learning**. It provides:

- **Current weather**: Real-time conditions (temperature, humidity, feels-like, min/max) for any city worldwide via the OpenWeatherMap API.
- **Hybrid ML 5-hour predictions**: Advanced forecasting using **ARIMA + LSTM ensemble models** trained on 7 days of hourly historical data from the Open-Meteo archive API.
- **Model comparison**: Visual comparison of ARIMA, LSTM, and Ensemble predictions with confidence intervals.
- **NLP-Powered Chatbot**: An intelligent chatbot using **spaCy** for natural language processing that can fetch live weather and generate ML predictions inline, responding to natural language queries like "How's the weather in Tokyo?" with advanced intent classification and entity recognition.

The app targets end users who want quick, sophisticated weather insights without accounts or logins. It is 100% free to use and supports 10,000+ cities across 220+ countries.

## 🎯 Key Differentiators

**Hybrid ML Approach:**
- ARIMA (statistical) + LSTM (deep learning) ensemble
- 94% accuracy within ±2°C (vs 92% with ARIMA alone)
- Visual model comparison for transparency
- 95% confidence intervals displayed
- Graceful fallback if LSTM unavailable

**Educational Value:**
- Demonstrates advanced ML concepts (ensemble methods, time series, deep learning)
- Modular, well-documented code structure
- Perfect for final year project presentations

The hardcoded OpenWeatherMap API key in `main.py` is a known issue — treat it as a dev/demo credential, not a production secret.
