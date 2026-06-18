# Tech Stack

## Backend
- **Python 3.8+** (developed/tested on 3.12)
- **Flask 2.2.3** — web framework, Jinja2 templating
- **Werkzeug 2.2.3** — WSGI utilities
- **requests 2.28.2** — HTTP calls to OpenWeatherMap and Open-Meteo APIs

## Data & ML
- **pandas ≥ 2.0** — data manipulation, CSV I/O
- **numpy ≥ 1.24, < 2.0** — numerical computing (pinned below 2.0 for pmdarima compatibility)
- **pmdarima ≥ 2.0.4** — `auto_arima` for automatic ARIMA order selection
- **statsmodels ≥ 0.14** — `ARIMA` model fitting and forecasting
- **scikit-learn ≥ 1.3** — ML utilities
- **matplotlib ≥ 3.7** — chart generation (uses `Agg` non-interactive backend)
- **spacy ≥ 3.7** — NLP for chatbot intent classification and entity recognition (requires `en_core_web_sm` model)

## Frontend
- **Bootstrap 5.2.3** — UI components and layout (CDN)
- **Chart.js** — temperature and humidity line charts (CDN)
- **Font Awesome** — icons (CDN via kit)
- **Google Fonts (Nunito)** — typography
- Vanilla JavaScript for chatbot widget and AQ form interactions
- **NLP Backend**: spaCy-powered chatbot with intent classification and entity recognition

## Production
- **gunicorn 20.1.0** — WSGI server for deployment

## External APIs
- **OpenWeatherMap** (`api.openweathermap.org/data/2.5/weather`) — current weather; API key hardcoded in `main.py`
- **Open-Meteo Archive** (`archive-api.open-meteo.com/v1/archive`) — 7-day hourly historical data; no API key required

## Common Commands

```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run development server
python main.py
# App available at http://127.0.0.1:5000

# Fix numpy/pmdarima compatibility if needed
pip install "numpy<2.0.0" --force-reinstall

# Install spaCy NLP model for chatbot
python -m spacy download en_core_web_sm

# Production server
gunicorn main:app
```

## Notes
- `matplotlib` must use the `Agg` backend (`matplotlib.use('Agg')`) since there is no display in a server context.
- Intermediate weather data is written to `static/csv/weather_data.csv` during prediction requests.
- No test framework is currently set up.
- No build step — pure Python/HTML/CSS/JS, no bundler or transpiler.
