# PROJECT_CONTEXT.md — ClimaSense Consolidated Reference

> Single source of truth for the ClimaSense project. Consolidates all documentation into one structured file.

---

## 1. Project Overview

ClimaSense is a Flask-based smart weather monitoring web application with **Hybrid Machine Learning** (ARIMA + LSTM ensemble). It provides current weather, 5-hour ML predictions, model comparison visualizations, a spaCy-powered NLP chatbot, and weather alert/dashboard features.

### Key Differentiators
- **Hybrid ML Approach**: ARIMA (statistical) + LSTM (deep learning) ensemble — 94% accuracy within ±2°C (vs 92% with ARIMA alone).
- **Model Transparency**: Visual comparison of ARIMA, LSTM, and Ensemble predictions with 95% confidence intervals.
- **NLP-Powered Chatbot**: spaCy-based intent classification and entity recognition; handles natural phrasing like "How's the weather in Tokyo?" — 93% accuracy (up from 44% keyword-matching baseline).
- **Weather Alerts & Dashboard**: Contextual weather alerts and live ML performance metrics displayed after each prediction.
- **Global Coverage**: Supports 10,000+ cities across 220+ countries. 100% free to use — no login required.

### Target Audience
End users who want quick, sophisticated weather insights. Also an educational resource demonstrating advanced ML concepts in a full-stack app.

### Version History
- **v1.0**: ARIMA-only prediction, basic keyword chatbot.
- **v2.0** (current): Hybrid ARIMA + LSTM ensemble, spaCy NLP chatbot, weather alerts, ML dashboard, database architecture, accuracy baseline adjustment.

---

## 2. Tech Stack & Architecture

### Backend
| Technology | Version | Purpose |
|---|---|---|
| Python | 3.8+ (tested on 3.12) | Core language |
| Flask | 2.2.3 | Web framework, Jinja2 templating |
| Werkzeug | 2.2.3 | WSGI utilities |
| requests | 2.28.2 | HTTP calls to external APIs |
| gunicorn | 20.1.0 | Production WSGI server |

### Data & ML
| Technology | Version | Purpose |
|---|---|---|
| pandas | ≥2.0 | Data manipulation, CSV I/O |
| numpy | ≥1.24, <2.0 | Numerical computing (pinned below 2.0 for pmdarima compatibility) |
| pmdarima | ≥2.0.4 | `auto_arima` for ARIMA order selection |
| statsmodels | ≥0.14 | ARIMA model fitting and forecasting |
| scikit-learn | ≥1.3 | ML utilities |
| matplotlib | ≥3.7 | Chart generation (Agg non-interactive backend) |
| tensorflow/keras | ≥2.15/≥3.0 | LSTM deep learning (optional; system degrades gracefully without it) |
| spacy | ≥3.7 | NLP for chatbot (requires `en_core_web_sm`) |

### Frontend
| Technology | Purpose |
|---|---|
| Bootstrap 5.2.3 (CDN) | UI components and layout |
| Chart.js (CDN) | Temperature, humidity, and model comparison charts |
| Font Awesome (CDN) | Icons |
| Google Fonts — Nunito (CDN) | Global typography |
| Vanilla JavaScript | Chatbot widget, AQ form interactions |

### External APIs
| API | Purpose | Cost |
|---|---|---|
| OpenWeatherMap (`api.openweathermap.org/data/2.5/weather`) | Current weather; API key hardcoded in `main.py` (dev/demo only) | Free tier |
| Open-Meteo Archive (`archive-api.open-meteo.com/v1/archive`) | 7-day hourly historical data (no API key required) | Free |

### System Architecture
```
User Browser (HTML/CSS/JS/Chart.js)
        ↓ HTTP Requests
Flask Web Server (main.py)
  ├── Routes Handler
  ├── Chatbot Logic  (chatbot_nlp.py)
  └── Prediction Engine (ml_models/)
        ↓
  ┌──────────────────┬──────────────────┬──────────────────┐
  │ OpenWeatherMap   │  Open-Meteo API  │  Hybrid ML Model │
  │ (Current Data)   │  (Historical)    │  (ARIMA + LSTM)  │
  └──────────────────┴──────────────────┴──────────────────┘
```

**Data flow:** User → Flask → APIs → ML Models → Results → User

### Project File Structure
```
ClimaSense/
├── main.py                        # Flask routes, ML logic, chatbot endpoint
├── chatbot_nlp.py                 # NLP chatbot (spaCy intent & entity recognition)
├── database.py                    # SQLite database module (optional integration)
├── requirements.txt               # Pinned Python dependencies
│
├── templates/
│   ├── index.html                 # Main page: weather, prediction, charts, chatbot
│   └── 404_error.html             # Error page for invalid city lookups
│
├── static/
│   ├── css/
│   │   ├── styles.css             # Global styles
│   │   ├── chatbot.css            # Chatbot widget styles
│   │   ├── dashboard.css          # Dashboard & alerts styles
│   │   ├── model_comparison.css   # Model comparison section styles
│   │   └── error_css.css          # 404 error page styles
│   ├── csv/
│   │   └── weather_data.csv       # Runtime-generated; ephemeral per request
│   └── [images]                   # Background images, favicon
│
├── ml_models/
│   ├── arima_model.py             # ARIMAPredictor class
│   ├── lstm_model.py              # LSTMPredictor class
│   ├── ensemble_model.py          # EnsemblePredictor class
│   ├── __init__.py
│   └── README.md                  # ML model documentation (preserved)
│
├── install_nlp_chatbot.sh/.bat    # spaCy NLP installers
└── test_nlp_chatbot.py            # NLP chatbot test suite
```

### Key Conventions
- **Single-page frontend**: `index.html` is the only active template; uses Jinja2 conditionals (`{% if status %}`, `{% if predict_status %}`) to toggle states.
- **Modular backend**: Flask logic in `main.py`, NLP in `chatbot_nlp.py`, ML in `ml_models/`.
- **Static assets via CDN**: Do not add local copies of Bootstrap, Chart.js, Font Awesome, or Google Fonts.
- **`weather_data.csv` is ephemeral**: Overwritten on every prediction request — not persistent storage.
- **matplotlib backend**: Must use `matplotlib.use('Agg')` (no display in server context).
- **`{{ variable }}` in `index.html`**: Jinja2 syntax, not JavaScript — VS Code may flag as errors (false positive).

---

## 3. ML Models (ARIMA, LSTM, Ensemble)

### Overview
ClimaSense uses a Hybrid ML approach combining three forecasting components trained on 7 days (168 hours) of hourly historical data from Open-Meteo.

```
Historical Weather Data (7 days, hourly)
              ↓
    ┌─────────┴──────────┐
    ↓                    ↓
ARIMA Model         LSTM Model
(Statistical)       (Deep Learning)
MAE: ~1.5°C         MAE: ~1.4°C (optimized)
    ↓                    ↓
    └─────────┬──────────┘
              ↓
   Ensemble Predictor
   (60% ARIMA + 40% LSTM)
   MAE: ~1.2°C  |  ~96% accuracy
              ↓
   5-Hour Forecast + Confidence Intervals
```

### ARIMA Model (`ml_models/arima_model.py`)
- **Type**: AutoRegressive Integrated Moving Average (statistical time series)
- **Auto-tuning**: `auto_arima` with `max_p=5, max_q=5, max_d=2`, `n_fits=50`, AIC criterion, ADF stationarity test
- **Fitting**: `method='lbfgs'`, `maxiter=500`, `enforce_stationarity=False`, `enforce_invertibility=False`
- **Separate models** for temperature and humidity
- **Native 95% confidence intervals**
- **Metrics tracked**: MAE, RMSE, AIC, BIC
- **Training time**: ~3–4 seconds
- **Best for**: Stable, predictable weather patterns

Example usage:
```python
from ml_models.arima_model import ARIMAPredictor
predictor = ARIMAPredictor()
training_info = predictor.train(temperature_data, humidity_data)
temp_pred, hum_pred, confidence = predictor.predict(steps=5)
```

### LSTM Model (`ml_models/lstm_model.py`)
- **Type**: Long Short-Term Memory neural network (deep learning)
- **Architecture**: 3 LSTM layers (100, 100, 50 units) + recurrent dropout (0.1) + Dropout (0.3, 0.3, 0.2) + Dense layers (50→25→1)
- **Loss function**: Huber (robust to outliers)
- **Optimizer**: Adam (lr=0.001)
- **Training**: 100 epochs, batch_size=16, validation_split=0.15, EarlyStopping (patience=15), ReduceLROnPlateau
- **Lookback window**: 24 hours
- **Requires TensorFlow ≥2.15 / Keras ≥3.0**; gracefully unavailable otherwise
- **Training time**: ~12–15 seconds
- **Best for**: Complex, non-linear weather patterns

Example usage:
```python
from ml_models.lstm_model import LSTMPredictor
predictor = LSTMPredictor(lookback=24)
training_info = predictor.train(temperature_data, humidity_data, epochs=100)
temp_pred, hum_pred, confidence = predictor.predict(temperature_data, humidity_data, steps=5)
```

### Ensemble Model (`ml_models/ensemble_model.py`)
- **Method**: Weighted averaging — default 60% ARIMA + 40% LSTM
- **Rationale**: ARIMA is more stable for weather; LSTM adds value for complex trends; 60/40 optimized through testing
- **Fallback**: Automatically uses ARIMA-only if LSTM is unavailable
- **Prediction caching** and error calculation utilities included
- **Training time**: ~15–19 seconds total

Example usage:
```python
from ml_models.ensemble_model import EnsemblePredictor
ensemble = EnsemblePredictor(arima_weight=0.6, lstm_weight=0.4)
training_info = ensemble.train(temp_data, hum_data, lstm_epochs=100)
all_predictions = ensemble.predict(temp_data, hum_data, steps=5)
# Access: all_predictions['models']['arima'], ['lstm'], ['ensemble']
```

### Accuracy Improvements Summary

| Metric | ARIMA Only (original) | Optimized Ensemble | Improvement |
|---|---|---|---|
| Temperature MAE | 1.8°C | ~1.2°C | ~33% better |
| Humidity MAE | 4.2% | ~3.0% | ~29% better |
| Accuracy (±2°C) | 92% | ~96% | +4% |
| Confidence Intervals | No | Yes (95%) | New feature |

### Baseline Temperature Adjustment
After ensemble predictions are generated, a baseline correction is applied anchoring Hour 1 prediction to the current actual temperature (from OpenWeatherMap API):
```
adjustment = current_actual_temp - ml_prediction_now
Hour 1: prediction + (adjustment × 0.8)
Hour 2: prediction + (adjustment × 0.7)
Hour 3: prediction + (adjustment × 0.6)
Hour 4: prediction + (adjustment × 0.5)
Hour 5: prediction + (adjustment × 0.4)
```
This technique is used in professional forecasting to bridge historical-pattern models with real-time conditions.

### Model Comparison Reference

| Feature | ARIMA | LSTM | Ensemble |
|---|---|---|---|
| Type | Statistical | Deep Learning | Hybrid |
| Training Time | 3–4s | 12–15s | 15–19s |
| Temp MAE | ~1.5°C | ~1.4°C | ~1.2°C |
| Humidity MAE | ~3.5% | ~3.2% | ~3.0% |
| Accuracy (±2°C) | 92% | ~89% | ~96% |
| Confidence Intervals | Native | Estimated | Combined |
| Interpretability | High | Low | Medium |
| Overfitting Risk | Low | Medium | Low |
| TensorFlow Required | No | Yes | No (fallback) |

### Dependencies Note
- `numpy<2.0` is **required** — pmdarima has a known incompatibility with NumPy 2.0+ (`AttributeError: np.NaN was removed`).
- TensorFlow/Keras are **optional** — without them, system falls back to ARIMA-only (still 92% accuracy).

---

## 4. NLP Chatbot

### Overview
The chatbot was upgraded from simple keyword matching to a spaCy-powered NLP system. It is implemented in `chatbot_nlp.py` and integrated into the `/chatbot` Flask endpoint.

**Performance comparison:**

| Metric | Keyword Matching (old) | spaCy NLP (current) |
|---|---|---|
| Overall Accuracy | ~44% | ~93% |
| Intent Classification | ~63% | ~91% |
| City Extraction | ~60% | ~95% |
| Natural Language Support | Limited | Full |
| Synonym Understanding | No | Yes |

### Architecture

```
User Message
     ↓
Flask App (main.py) — POST /chatbot
     ↓
NLPChatbot (chatbot_nlp.py)
     ↓
┌────────────────────┬────────────────────┐
│  IntentClassifier  │  EntityExtractor   │
│  (spaCy NLP)       │  (spaCy NER)       │
│  Lemmatization     │  GPE/LOC entities  │
│  POS tagging       │  Proper noun detect│
│  Confidence score  │  Pattern fallback  │
└────────────────────┴────────────────────┘
     ↓
Response Generator → Weather APIs (if needed) → JSON response
```

### Components

#### `IntentClassifier`
Scores each message against 13 supported intents using lemmatized tokens:
- `weather_current` — "What's the weather in London?"
- `weather_forecast` — "Give me the forecast for Tokyo"
- `greeting` — "Hello", "Hi"
- `about` — Information about ClimaSense
- `help` — How to use
- `accuracy` — "How accurate are your predictions?" / "Are you reliable?"
- `technology` — ML/AI questions
- `location` — Coverage questions
- `cost` — Pricing questions
- `rain` — Rain-specific queries
- `humidity` — Humidity queries
- `thank` — Thank you messages
- `goodbye` — Farewell messages

#### `EntityExtractor`
Extracts city names using:
1. spaCy NER — looks for GPE (Geopolitical Entity) and LOC entities
2. Proper noun detection near weather keywords
3. Pattern matching fallback (legacy keyword patterns)

#### `NLPChatbot`
Main orchestrator:
- Calls `IntentClassifier` and `EntityExtractor`
- Routes to weather API calls or static responses
- Maintains `conversation_history` (list of `{message, intent, city, timestamp}`)
- Exposes `get_response(message) → (response_str, quick_replies_list)`

#### Singleton factory
```python
from chatbot_nlp import get_chatbot
chatbot = get_chatbot()  # Reuses single instance
```

### Graceful Fallback
If spaCy is not installed or `en_core_web_sm` is not found, the chatbot automatically falls back to keyword-matching mode. Check availability:
```python
from chatbot_nlp import SPACY_AVAILABLE
print(SPACY_AVAILABLE)  # True or False
```

### Flask Integration
```python
from chatbot_nlp import get_chatbot

@app.route('/chatbot', methods=['POST'])
def chatbot():
    nlp_chatbot = get_chatbot()
    user_message = request.json.get('message', '')
    response, quick_replies = nlp_chatbot.get_response(user_message)
    return jsonify({'success': True, 'response': response, 'quick_replies': quick_replies})
```

### spaCy Model Options
| Model | Size | Speed | Accuracy | Notes |
|---|---|---|---|---|
| `en_core_web_sm` | ~13 MB | Fast | Good | **Default — recommended** |
| `en_core_web_md` | ~43 MB | Medium | Better | Includes word vectors |
| `en_core_web_lg` | ~560 MB | Slower | Best | Full word vectors |

To switch model, update `chatbot_nlp.py`:
```python
nlp = spacy.load("en_core_web_md")
```

### Performance Characteristics
- Intent detection: ~15ms
- Entity extraction: ~10ms
- Response generation: ~5ms
- **Total NLP overhead: ~30ms** (acceptable)
- Memory footprint: ~13–14 MB (one-time load for spaCy model)

### Query Examples That Now Work
```
✅ "How's the weather in London?"
✅ "Tell me about Mumbai's weather"
✅ "What's it like in Tokyo right now?"
✅ "Give me the forecast for New York"
✅ "Is it raining in Paris?"
✅ "Are your predictions reliable?"
✅ "What ML model do you use?"
✅ "Which cities do you support?"
```

### Installation
```bash
# Automated
./install_nlp_chatbot.sh     # Linux/Mac
install_nlp_chatbot.bat      # Windows

# Manual
pip install "spacy>=3.7.0"
python -m spacy download en_core_web_sm

# Verify
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✅ OK')"
```

### Testing
```bash
python test_nlp_chatbot.py
# Expected: 19/19 tests pass, 100% success rate
```

### Future Enhancement Ideas
1. Multi-language support (Spanish, French, German spaCy models)
2. Context memory — multi-turn conversations ("What about tomorrow?")
3. Sentiment analysis for tone-adjusted responses
4. Custom NER training for weather-specific entities

---

## 5. Frontend & UI

### Single-Page Application
`index.html` is the only active template (557+ lines). It uses Jinja2 conditionals to toggle between empty and filled states:
- `{% if status %}` — shows current weather results
- `{% if predict_status %}` — shows prediction section (table, charts, alerts, dashboard)

### Page Sections (in order)
1. **Hero Section** — Title, tagline, background image (`weatherbg.png`)
2. **Feature Navigation Cards** — Weather, Prediction, AQI placeholders
3. **Current Weather Section** — Search form + results card (temp, feels like, min/max, humidity, description)
4. **Weather Prediction Section** — City input + "Predict" button
   - 5-hour prediction table (temperature & humidity)
   - Temperature trend chart (Chart.js line)
   - Humidity trend chart (Chart.js line)
   - Rain prediction text (if humidity ≥ 75%)
5. **Weather Alerts Section** *(shows after prediction, pink gradient)* — dynamically generated alert cards
6. **ML Performance Dashboard** *(shows after prediction, purple gradient)* — metrics, model cards, confidence
7. **Chatbot Widget** — Fixed bottom-right, toggleable chat window, quick-reply buttons

### CSS Files
| File | Purpose |
|---|---|
| `static/css/styles.css` | Global layout, sections, table, general styles |
| `static/css/chatbot.css` | Chatbot widget, bubbles, quick replies |
| `static/css/dashboard.css` | Dashboard metrics cards and alert cards |
| `static/css/model_comparison.css` | ML model comparison cards and chart container |
| `static/css/error_css.css` | 404 error page |

**Rule**: Add new CSS to appropriate existing file — do not create new stylesheets unless truly separate.

### Chart.js Visualizations
- **Temperature Chart**: Line chart, red, shows 5-hour forecast; labels from Flask (`tlabels`, `tvalues`)
- **Humidity Chart**: Line chart, shows 5-hour humidity; labels from Flask
- **Model Comparison Chart** (when hybrid mode active):
  - Red solid line: Ensemble prediction
  - Blue dashed line: ARIMA only
  - Green dashed line: LSTM only (if available)
- All charts: `responsive: true`, interactive tooltips

### Responsive Design Breakpoints (Bootstrap 5)
- Desktop (>992px): 3-column grid for metric cards and model cards
- Tablet (768–992px): 2-column grid
- Mobile (<768px): Single-column stacked, horizontal scroll for tables

### Visual Design Language
- **Font**: Nunito (Google Fonts CDN) — global font family
- **Dashboard gradient**: Purple `#667eea → #764ba2`
- **Alerts gradient**: Pink `#f093fb → #f5576c`
- **Cards**: White, `border-radius: 15px`, box-shadow, hover lift (+5px) with 0.3s transition
- **Critical alerts**: Pulse animation (2s loop) for danger-level alerts
- **Alert entry**: Slide-down animation (0.5s)

### Alert Types and Colors
| Alert | Color | Condition |
|---|---|---|
| Pleasant Weather | Green (success) | Default / good conditions |
| Informational | Blue (info) | Dry weather (humidity <30%) |
| Warning | Orange | Temp rising/dropping >5°C, rain possible (humidity ≥75%) |
| Danger | Red | Heat >38°C, cold <5°C, heavy rain (humidity ≥85%) |

### ML Dashboard Components (after prediction)
- **4 metric cards**: Data Points (192), Models Trained (1–2), Accuracy (96%), Training Time (15–20s)
- **Model comparison cards**: ARIMA card (order, MAE, RMSE, AIC), LSTM card (lookback, loss, MAE, epochs), Ensemble card (ARIMA weight 60%, LSTM weight 40%, method, expected accuracy)
- **Confidence section**: 95% confidence badge, ±X°C range, model type

### JavaScript Notes
- Chatbot widget: Pure vanilla JS in `index.html`
- Chart.js data injected via Jinja2 `{{ tlabels | tojson }}` / `{{ tvalues | tojson }}`
- No build step — no bundler or transpiler

### Page Layout Visual
```
BEFORE (v1): Hero → Feature Cards → Current Weather → Prediction (table + charts + rain)
AFTER  (v2): Hero → Feature Cards → Current Weather → Prediction (table + charts + rain)
             → Weather Alerts (pink) → ML Dashboard (purple) → [Chatbot widget always visible]
```

---

## 6. Database

### Overview (`database.py`)
An optional SQLite database module providing persistence for historical data caching, prediction tracking, and analytics. Not active by default — requires integration into `main.py`.

### Schema
```
cities (id, name, country, latitude, longitude, created_at)
    1:N ↓
    ├── historical_weather (id, city_id, timestamp, temperature, humidity, feels_like,
    │                       pressure, wind_speed, description, fetched_at)
    │   INDEX on (city_id, timestamp)
    │
    ├── predictions (id, city_id, prediction_time, target_time,
    │               predicted_temp, predicted_humidity,
    │               confidence_lower, confidence_upper, model_used,
    │               actual_temp, actual_humidity, error_temp, error_humidity, created_at)
    │   INDEX on (city_id, target_time)
    │
    ├── model_performance (id, city_id, model_name, data_points,
    │                      temperature_mae, temperature_rmse, humidity_mae, humidity_rmse,
    │                      training_time, metadata [JSON], created_at)
    │
    └── user_queries (id, city_name, query_type, ip_address, user_agent,
                      response_time, success, created_at)
        INDEX on (created_at)
```

### Key API Methods
```python
from database import get_database
db = get_database()

# Cities
city_id = db.add_city(name, country, latitude, longitude)
city    = db.get_city(name, country)

# Historical data
db.add_historical_data(city_id, records_list)   # records: [{timestamp, temperature, humidity, ...}]
df = db.get_historical_data(city_id, hours=168)  # Returns pandas DataFrame
has_data = db.has_recent_data(city_id, hours=24)

# Predictions
pred_id = db.save_prediction(city_id, prediction_dict)
db.update_prediction_actual(pred_id, actual_temp, actual_humidity)
accuracy = db.get_prediction_accuracy(city_id, days=7)

# Model performance
db.save_model_performance(city_id, model_perf_dict)
history = db.get_model_performance_history(city_id, 'ARIMA', limit=10)

# Analytics
db.log_user_query(city_name, query_type, response_time, success, ip_address, user_agent)
popular = db.get_popular_cities(limit=10)
stats   = db.get_query_stats(days=7)
db_stats = db.get_database_stats()

# Maintenance
db.clear_old_data(days=30)
db.close()
```

### Caching Strategy
```
Request → Check DB for city data (< 24 hours old)?
  YES → Use cached historical data (saves ~12 seconds API fetch)
  NO  → Fetch from Open-Meteo API → Save to DB → Use data
```
This makes subsequent requests for the same city ~60% faster.

### Benefits
- Avoids repeated API calls for the same city
- Tracks prediction accuracy over time (save prediction, update with actuals later)
- Analytics: popular cities, peak usage, success rates, performance trends
- `.gitignore` should include `weather_data.db` and `*.db-journal`

### Data Retention Recommendations
- Historical weather: 30 days
- Predictions: 30 days (for accuracy tracking)
- User queries: 90 days (for analytics)
- Model performance: Keep all (small dataset)

---

## 7. Climate Analytics

### Data Sources
- **Current weather**: OpenWeatherMap API — real-time temperature, humidity, feels-like, min/max, description, country
- **Historical data**: Open-Meteo Archive API — 7 days (168 hours) of hourly `temperature_2m` and `relative_humidity_2m`

### Data Collection & Processing
```python
# Coordinates from OpenWeatherMap
LAT = data['coord']['lat']
LON = data['coord']['lon']

# Open-Meteo fetch
end_date   = datetime.now() - timedelta(days=2)
start_date = end_date - timedelta(days=7)
params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "hourly": "temperature_2m,relative_humidity_2m",
    "timezone": "auto"
}
```

### Data Requirements
- **Minimum**: 24 data points for ARIMA
- **Recommended**: 168 data points (7 days)
- **Optimal**: 720 data points (30 days)
- Null values are filtered out before training

### Open-Meteo Data Quality
- Source: ERA5 reanalysis (ECMWF)
- Temperature accuracy: within ~2°C
- Resolution: hourly granularity
- Availability: global coverage
- Cost: Free, no API key

### Prediction Accuracy Expectations
| Horizon | ClimaSense | Google/Commercial |
|---|---|---|
| 1 hour | ±1–2°C | ±0.5–1°C |
| 2 hours | ±2–3°C | ±1–2°C |
| 3 hours | ±3–4°C | ±2–3°C |
| 4 hours | ±4–5°C | ±3–4°C |
| 5 hours | ±5–6°C | ±4–5°C |

ClimaSense uses 192 hourly historical data points and real-time temperature anchoring. Google uses real-time satellite data, atmospheric pressure models, and supercomputer models — the difference is expected and honest.

### Intermediate Storage
`static/csv/weather_data.csv` — written during each prediction request, then read by ML models. This file is **ephemeral** (overwritten each request). Not for persistent storage.

### Supported Cities
- 10,000+ cities across 220+ countries
- Any city supported by OpenWeatherMap works

---

## 8. Dashboard & Alerts

### Weather Alerts System (`generate_weather_alerts()` in `main.py`)
Analyzes the 5-hour predictions and generates contextual alert cards. Displayed in a pink-gradient section after prediction.

**Alert logic:**
| Alert | Type | Condition |
|---|---|---|
| Temperature Rising | warning | Temp increase >5°C over 5 hours |
| Temperature Dropping | warning | Temp decrease >5°C over 5 hours |
| Heat Alert | danger | Any predicted temp >38°C |
| Cold Alert | danger | Any predicted temp <5°C |
| Heavy Rain Expected | danger | Max predicted humidity ≥85% |
| Rain Possible | warning | Max predicted humidity ≥75% |
| Dry Weather | info | Max predicted humidity <30% |
| Pleasant Weather | success | Default — none of the above |

Each alert includes: `{type, icon (Font Awesome class), title, message, time}`.

**Adding a custom alert type** (in `main.py`):
```python
if some_condition:
    alerts.append({
        'type': 'warning',         # success | info | warning | danger
        'icon': 'fa-wind',
        'title': 'High Wind Warning',
        'message': 'Strong winds expected.',
        'time': datetime.now().strftime('%I:%M %p')
    })
```

### ML Performance Dashboard (`prepare_dashboard_metrics()` in `main.py`)
Displayed in a purple-gradient section after the alerts. Shows live metrics computed during prediction.

**Metric cards:**
- **Data Points**: Number of historical hours analyzed (typically 192)
- **Models Trained**: Count of successfully trained models (1 ARIMA-only, or 2 with LSTM)
- **Accuracy**: Displayed as percentage (e.g., 96%)
- **Training Time**: e.g., "15–20 seconds"

**Model comparison cards:**
- ARIMA card: temperature order (p,d,q), temp MAE, temp RMSE, AIC, humidity MAE
- LSTM card (when available): lookback, temp loss, temp MAE, epochs trained, humidity loss
- Ensemble card: ARIMA weight (60%), LSTM weight (40%), method (Weighted Average), expected accuracy

**Confidence section:**
- 95% confidence badge
- Confidence range (±X°C computed from prediction spread)
- Model type string

### Visual Design
- **Dashboard**: Purple gradient background, white cards, Font Awesome icons
- **Alerts**: Pink gradient background, color-coded cards by severity
- **Hover effects**: Cards lift 5px with 0.3s transition
- **Critical alerts**: Red pulse animation (2-second loop) for danger-type alerts
- All in `static/css/dashboard.css`

### Presentation Demo Flow
1. Homepage → search city for current weather
2. Click "Predict" — explain: "Training on 192 hours of data"
3. Wait ~15–20 seconds
4. Show prediction table + charts
5. Scroll to **Weather Alerts** — highlight severity colors, explain they're computed from predictions
6. Scroll to **ML Dashboard** — explain:
   - 192 data points, 2 models trained
   - ARIMA MAE 0.33°C, ensemble weighting 60/40
   - 95% confidence interval meaning
7. Demo chatbot: "What's the weather in London?"

---

## 9. Setup & Installation

### Prerequisites
- Python 3.8+ (developed/tested on 3.12)
- Internet connection (APIs needed at runtime)
- 4 GB RAM minimum (8 GB recommended)
- 500 MB disk space (excluding virtual environment)

### Quick Start
```bash
# 1. Clone the repository
git clone https://github.com/DeboFTW/ClimaSense.git
cd ClimaSense

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Fix numpy/pmdarima compatibility if needed
pip install "numpy<2.0.0" --force-reinstall

# 5. Run development server
python main.py
# App available at http://127.0.0.1:5000
```

### Optional: LSTM (Deep Learning) Support
```bash
pip install tensorflow>=2.15.0 keras>=3.0.0
# Without this, system uses ARIMA-only (92% accuracy, fully functional)
```

### Optional: NLP Chatbot (spaCy)
```bash
# Automated (recommended):
./install_nlp_chatbot.sh    # Linux/Mac
install_nlp_chatbot.bat     # Windows

# Manual:
pip install "spacy>=3.7.0"
python -m spacy download en_core_web_sm

# Verify:
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✅ spaCy OK')"
```

### Production Deployment
```bash
gunicorn main:app
# Or with workers:
gunicorn -w 4 -b 0.0.0.0:8000 main:app
```

Production checklist:
- [ ] Change `debug=False` in `main.py`
- [ ] Move API key to environment variable
- [ ] Add rate limiting middleware
- [ ] Add CSRF token protection
- [ ] Consider Redis caching for API responses

### Common Troubleshooting

| Problem | Solution |
|---|---|
| `python` not found | Use `python3 main.py` |
| Port 5000 in use | `pkill -9 python` (Mac/Linux) or `taskkill /F /IM python.exe` (Windows) |
| `Module not found` | `pip install -r requirements.txt` |
| NumPy compatibility error | `pip install "numpy<2.0.0" --force-reinstall` |
| City not found | Check spelling; try "Mumbai, IN" or "Paris, FR" |
| Slow predictions | Normal — LSTM takes 12–15s; ARIMA 3–4s; total ~20s |
| VS Code errors in `index.html` | False positives — `{{ }}` is Jinja2, not JS |
| spaCy model not found | `python -m spacy download en_core_web_sm` |
| LSTM shows as disabled | Install TensorFlow: `pip install tensorflow` |

### Running Tests
```bash
# NLP chatbot tests
python test_nlp_chatbot.py
# Expected: 19/19 pass

# No formal test framework is set up for the rest of the app (future enhancement)
```

### Stopping the Server
```bash
Ctrl+C             # Stop Flask server
deactivate         # Exit virtual environment
```

---

## 10. API & Integration Notes

### External API Reference

#### OpenWeatherMap
- **Endpoint**: `https://api.openweathermap.org/data/2.5/weather`
- **Parameters**: `q={city}`, `appid={API_KEY}`, `units=metric`
- **Response fields used**: `main.temp`, `main.feels_like`, `main.temp_min`, `main.temp_max`, `main.humidity`, `weather[0].description`, `sys.country`, `name`, `coord.lat`, `coord.lon`
- **API key**: Hardcoded in `main.py` — treat as dev/demo credential; use environment variable for production
- **Free tier**: 60 calls/min, 1,000,000 calls/month
- **Error handling**: Non-200 or `KeyError` → redirect to `404_error.html`

#### Open-Meteo Archive
- **Endpoint**: `https://archive-api.open-meteo.com/v1/archive`
- **Parameters**: `latitude`, `longitude`, `start_date`, `end_date`, `hourly=temperature_2m,relative_humidity_2m`, `timezone=auto`
- **No API key required**
- **Date range**: `end_date = now - 2 days`, `start_date = end_date - 7 days`
- **Data source**: ERA5 reanalysis (ECMWF)
- **Null handling**: Filter out `None` values before passing to ML models

### Internal Flask Endpoints

| Route | Method | Purpose | Response |
|---|---|---|---|
| `/` | GET/POST | Main dashboard, current weather search | HTML page |
| `/predict-weather` | POST | Generate 5-hour ML predictions | HTML page (Jinja2 rendered) |
| `/chatbot` | POST | NLP chatbot API | JSON `{success, response, quick_replies}` |

#### `/chatbot` Request/Response
```json
// Request (application/json)
{ "message": "What's the weather in Mumbai?" }

// Response
{
  "success": true,
  "response": "🌍 Mumbai, IN ☀️\n🌡️ Temperature: 32°C\n💧 Humidity: 78%\n...",
  "quick_replies": ["Forecast for Mumbai", "Weather in Delhi", "How to use?"]
}
```

### Template Variables Passed to `index.html`
Key variables passed from Flask to the Jinja2 template after a prediction:

| Variable | Type | Description |
|---|---|---|
| `predict_status` | bool | Whether a prediction was made |
| `tvalues` / `hvalues` | list | 5-hour temperature / humidity predictions |
| `tlabels` | list | Time labels (e.g., `["14:30", "15:30", ...]`) |
| `model_used` | str | "ensemble" or "arima" |
| `lstm_available` | bool | Whether LSTM trained successfully |
| `arima_tvalues` / `lstm_tvalues` | list | Individual model predictions |
| `temp_confidence_lower/upper` | list | 95% confidence bounds |
| `training_info` | dict | Full training metrics dict |
| `weather_alerts` | list | Generated alert dicts |
| `dashboard_metrics` | dict | Dashboard metric values |

### Adding New ML Models to the Ensemble
1. Create `ml_models/new_model.py` with `train()` and `predict()` methods
2. Return predictions in standard format: `{'temperature': [...], 'humidity': [...]}`
3. Import and integrate in `ml_models/ensemble_model.py`
4. Update weights in `EnsemblePredictor.__init__()`

### Deployment Options
| Platform | Command / Notes |
|---|---|
| Local | `python main.py` → http://127.0.0.1:5000 |
| Production | `gunicorn main:app` |
| Heroku | Free tier available |
| AWS | EC2 + Elastic Beanstalk |
| Azure | App Service |
| Docker (future) | `FROM python:3.12-slim`, `COPY . /app`, `RUN pip install -r requirements.txt`, `CMD ["python", "main.py"]` |

### Security Notes
- API key is hardcoded in `main.py` — known dev/demo credential, not production secret
- For production: store in environment variable, add `.env` to `.gitignore`
- No user data stored (stateless, no auth)
- No SQL injection risk (no database by default)
- HTTPS used for all external API calls
- Input sanitized by Flask auto-escaping in Jinja2

---

*End of PROJECT_CONTEXT.md — Last consolidated: June 2026*
