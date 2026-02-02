# 🌍 Forecasting PM2.5 Air Pollution in Quito

**Inteligencia Artificial II — Proyecto Final (Progreso 3)**

Universidad de las Américas (UDLA) | Febrero 2026

## 👥 Authors
- Diego Toscano
- Andrés Guamán

## 📋 Description
Time series forecasting of PM2.5 (fine particulate matter < 2.5 µm) air pollution levels in Quito, Ecuador, using data from the REMMAQ (Red Metropolitana de Monitoreo Atmosférico de Quito) collected from 2004 to 2025 across 9 monitoring stations.

## 🔮 Models
- **ARIMA(X)** — Seasonal autoregressive integrated moving average
- **MLForecast** — Machine Learning (Random Forest + XGBoost) with automated feature engineering
- **LSTM** — Long Short-Term Memory neural network

## 📊 Forecasting Horizons
- **Monthly (5–10 years)** — For municipal planning and policy
- **Hourly (10 days)** — For citizen use and health alerts

## 🚀 Streamlit App
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ia-pm25-forecast-quito.streamlit.app/)

## 📂 Repository Structure
```
├── DTAG_Proyecto_Progreso_3_IA_II.ipynb  # Main Jupyter Notebook
├── streamlit_app.py                       # Streamlit web application
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
├── data_monthly_full.csv                  # Processed monthly data
├── forecast_monthly_5yr.csv               # 5-year monthly forecasts
├── forecast_hourly_10days.csv             # 10-day hourly forecasts
├── model_comparison.csv                   # Model evaluation results
└── evaluation_monthly.csv                 # Test set predictions
```

## 📦 Data Source
[REMMAQ — Secretaría de Ambiente de Quito](http://datosambiente.quito.gob.ec/)

## 🛠️ Setup
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
