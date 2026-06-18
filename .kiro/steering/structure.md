# Project Structure

```
ClimaSense/
├── main.py                        # Flask routes, ARIMA ML logic, chatbot endpoint
├── chatbot_nlp.py                 # NLP-powered chatbot (spaCy-based intent & entity recognition)
├── requirements.txt               # Pinned Python dependencies (includes spaCy)
│
├── templates/                     # Jinja2 HTML templates
│   ├── index.html                 # Main page: weather search, prediction, charts, chatbot widget
│   └── 404_error.html             # Error page for invalid city lookups
│
├── static/
│   ├── css/
│   │   ├── styles.css             # Global styles (layout, sections, table, charts)
│   │   ├── chatbot.css            # Chatbot widget styles
│   │   └── error_css.css          # 404 error page styles
│   ├── csv/
│   │   └── weather_data.csv       # Runtime-generated; stores fetched historical data per prediction request
│   ├── clouds-3.jpg               # Background image for weather section
│   ├── weatherbg.png              # Hero section background
│   └── weather.png                # Favicon
│
├── ml_models/                     # Machine learning model modules
│   ├── arima_model.py             # ARIMA model implementation
│   ├── lstm_model.py              # LSTM neural network model
│   └── ensemble_model.py          # Hybrid ensemble predictor
│
├── install_nlp_chatbot.sh         # Linux/Mac installer for spaCy NLP
├── install_nlp_chatbot.bat        # Windows installer for spaCy NLP
├── test_nlp_chatbot.py            # Test suite for NLP chatbot
├── NLP_CHATBOT_UPGRADE.md         # Full documentation for NLP upgrade
├── CHATBOT_UPGRADE_QUICKSTART.md  # Quick start guide for NLP chatbot
│
├── IMDAA_merged_*.csv / *.nc      # Legacy/research dataset files (not used by the app at runtime)
├── PRESENTATION.md                # Project presentation notes
└── PROJECT_REPORT.md              # Academic/project report
```

## Key Conventions

- **Modular backend**: Core Flask logic in `main.py`, NLP chatbot in `chatbot_nlp.py`, ML models in `ml_models/` directory
- **Single-page frontend**: `index.html` is the only active template. It uses Jinja2 conditionals (`{% if status %}`, `{% if predict_status %}`) to toggle between empty/filled states.
- **NLP-powered chatbot**: Uses spaCy for intent classification and entity recognition. Falls back gracefully to keyword matching if spaCy unavailable.
- **Static assets via CDN**: Bootstrap, Chart.js, Font Awesome, and Google Fonts are loaded from CDN — do not add local copies.
- **CSS per feature**: Add new CSS to the appropriate existing file (`styles.css` for layout/sections, `chatbot.css` for chatbot, `error_css.css` for error pages) rather than creating new stylesheets.
- **`weather_data.csv` is ephemeral**: It is overwritten on every prediction request. Do not treat it as persistent storage.
- **Font**: The app uses `Nunito` (Google Fonts) as the global font family.
- **Template syntax note**: `{{ variable }}` blocks in `index.html` are Jinja2, not JavaScript — VS Code may flag them as errors inside `<script>` tags; this is a false positive.
