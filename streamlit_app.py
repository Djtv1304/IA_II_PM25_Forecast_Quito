"""
🌍 PM2.5 Air Quality Forecast — Quito, Ecuador
Streamlit Dashboard for REMMAQ Data Analysis & Forecasting

Authors: Diego Toscano & Andrés Guamán
Universidad de las Américas (UDLA) — Inteligencia Artificial II
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PM2.5 Forecast — Quito",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B4332;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stMetric > div {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    """Load all pre-computed data files."""
    data = {}

    # Monthly historical data
    if os.path.exists('data_monthly_full.csv'):
        data['monthly'] = pd.read_csv('data_monthly_full.csv', parse_dates=['datetime'])
    elif os.path.exists('data_monthly.csv'):
        data['monthly'] = pd.read_csv('data_monthly.csv', parse_dates=['datetime'])

    # Model comparison
    if os.path.exists('model_comparison.csv'):
        data['comparison'] = pd.read_csv('model_comparison.csv')

    # Monthly 5-year forecast
    if os.path.exists('forecast_monthly_5yr.csv'):
        data['forecast_monthly'] = pd.read_csv('forecast_monthly_5yr.csv', parse_dates=['ds'])

    # Hourly 10-day forecast
    if os.path.exists('forecast_hourly_10days.csv'):
        data['forecast_hourly'] = pd.read_csv('forecast_hourly_10days.csv', parse_dates=['ds'])

    # Evaluation data
    if os.path.exists('evaluation_monthly.csv'):
        data['evaluation'] = pd.read_csv('evaluation_monthly.csv', parse_dates=['ds'])

    # ARIMA forecasts
    if os.path.exists('forecast_arima.csv'):
        data['arima'] = pd.read_csv('forecast_arima.csv', parse_dates=['datetime'])

    # Hourly recent data
    if os.path.exists('data_hourly_recent.csv'):
        data['hourly_recent'] = pd.read_csv('data_hourly_recent.csv', parse_dates=['datetime'])

    return data

data = load_data()

# Station metadata
STATIONS = ['BELISARIO', 'CARAPUNGO', 'CENTRO', 'COTOCOLLAO',
            'EL CAMAL', 'GUAMANI', 'LOS CHILLOS', 'SAN ANTONIO', 'TUMBACO']

STATION_COORDS = {
    'BELISARIO': (-0.1806, -78.4878, 2835),
    'CARAPUNGO': (-0.0986, -78.4475, 2660),
    'CENTRO':    (-0.2200, -78.5128, 2820),
    'COTOCOLLAO': (-0.1075, -78.4972, 2739),
    'EL CAMAL':  (-0.2514, -78.5147, 2840),
    'GUAMANI':   (-0.3325, -78.5517, 3066),
    'LOS CHILLOS': (-0.3000, -78.4600, 2453),
    'SAN ANTONIO': (-0.0017, -78.4417, 2438),
    'TUMBACO':   (-0.2122, -78.4000, 2331),
}

WHO_GUIDELINE = 15  # µg/m³ annual
NECAA_24H = 50  # µg/m³ 24-hour

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Escudo_de_Quito.svg/120px-Escudo_de_Quito.svg.png", width=80)
    st.markdown("## ⚙️ Configuración")

    # Station selector
    selected_station = st.selectbox(
        "📍 Estación de Monitoreo",
        STATIONS,
        index=0,
        help="Seleccione una estación de la REMMAQ"
    )

    # Forecast horizon
    forecast_horizon = st.radio(
        "🔮 Horizonte de Pronóstico",
        ["Mensual (5 años)", "Horario (10 días)"],
        help="Mensual: planificación municipal | Horario: uso ciudadano"
    )

    st.markdown("---")
    st.markdown("### 📊 Acerca del Proyecto")
    st.markdown("""
    **Forecasting PM2.5 en Quito**

    Proyecto final de IA II — UDLA

    👥 Diego Toscano & Andrés Guamán

    📅 Dataset REMMAQ 2004–2025

    🔗 [Datos Abiertos](http://datosambiente.quito.gob.ec)
    """)

    st.markdown("---")
    st.markdown(f"📍 **{selected_station}**")
    if selected_station in STATION_COORDS:
        lat, lon, elev = STATION_COORDS[selected_station]
        st.markdown(f"🏔️ Elevación: **{elev:,} m.s.n.m.**")

# ============================================================
# MAIN CONTENT
# ============================================================
st.markdown('<p class="main-header">🌍 Pronóstico de Calidad del Aire — PM2.5</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Distrito Metropolitano de Quito | REMMAQ 2004–2025</p>', unsafe_allow_html=True)

# ============================================================
# TAB LAYOUT
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Pronóstico", "📊 Análisis Histórico", "🏆 Modelos", "🗺️ Mapa"])

# ============================================================
# TAB 1: FORECAST
# ============================================================
with tab1:
    st.markdown(f"### 🔮 Pronóstico para **{selected_station}**")

    if forecast_horizon == "Mensual (5 años)":
        st.markdown("*Horizonte: 60 meses — Para planificación municipal*")

        if 'monthly' in data and 'forecast_monthly' in data:
            # Historical
            hist = data['monthly'][data['monthly']['station'] == selected_station].sort_values('datetime')
            # Forecast
            fcast = data['forecast_monthly']
            if 'unique_id' in fcast.columns:
                fcast_st = fcast[fcast['unique_id'] == selected_station].sort_values('ds')
            else:
                fcast_st = pd.DataFrame()

            fig = go.Figure()

            # Historical line
            if 'pm25_ugm3' in hist.columns:
                fig.add_trace(go.Scatter(
                    x=hist['datetime'], y=hist['pm25_ugm3'],
                    mode='lines', name='Histórico',
                    line=dict(color='#2196F3', width=1.5),
                    opacity=0.7
                ))

            # Evaluation (test period)
            if 'evaluation' in data:
                eval_st = data['evaluation'][data['evaluation']['unique_id'] == selected_station]
                if not eval_st.empty and 'predicted_XGB' in eval_st.columns:
                    fig.add_trace(go.Scatter(
                        x=eval_st['ds'], y=eval_st['predicted_XGB'],
                        mode='lines', name='XGBoost (Test)',
                        line=dict(color='#FF5722', width=2, dash='dash')
                    ))

            # Future forecast
            if not fcast_st.empty:
                pred_col = [c for c in fcast_st.columns if 'predicted' in c.lower() or 'xgb' in c.lower()]
                if pred_col:
                    fig.add_trace(go.Scatter(
                        x=fcast_st['ds'], y=fcast_st[pred_col[0]],
                        mode='lines', name='Pronóstico 5 años',
                        line=dict(color='#4CAF50', width=2.5),
                        fill='tozeroy', fillcolor='rgba(76, 175, 80, 0.1)'
                    ))

            # WHO guideline
            fig.add_hline(y=WHO_GUIDELINE, line_dash="dot", line_color="red",
                         annotation_text="OMS Guía Anual (15 µg/m³)")

            fig.update_layout(
                title=f"PM2.5 Pronóstico Mensual — {selected_station}",
                xaxis_title="Fecha",
                yaxis_title="PM2.5 (µg/m³)",
                height=500,
                template="plotly_white",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Key metrics
            if 'pm25_ugm3' in hist.columns:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    last_val = hist['pm25_ugm3'].dropna().iloc[-1] if not hist['pm25_ugm3'].dropna().empty else None
                    st.metric("Último Valor Mensual", f"{last_val:.1f} µg/m³" if last_val else "N/A")
                with col2:
                    avg_2024 = hist[hist['datetime'].dt.year >= 2024]['pm25_ugm3'].mean()
                    st.metric("Promedio 2024-25", f"{avg_2024:.1f} µg/m³" if not np.isnan(avg_2024) else "N/A")
                with col3:
                    overall_avg = hist['pm25_ugm3'].mean()
                    st.metric("Promedio Histórico", f"{overall_avg:.1f} µg/m³" if not np.isnan(overall_avg) else "N/A")
                with col4:
                    above_who = (hist['pm25_ugm3'] > WHO_GUIDELINE).mean() * 100
                    st.metric("% Sobre OMS", f"{above_who:.0f}%")
        else:
            st.warning("⚠️ Datos de pronóstico mensual no disponibles. Ejecute el notebook primero.")

    else:  # Hourly forecast
        st.markdown("*Horizonte: 240 horas (10 días) — Para uso ciudadano*")

        if 'forecast_hourly' in data:
            fcast_h = data['forecast_hourly']
            if 'unique_id' in fcast_h.columns:
                fcast_h_st = fcast_h[fcast_h['unique_id'] == selected_station].sort_values('ds')
            else:
                fcast_h_st = pd.DataFrame()

            fig = go.Figure()

            # Recent historical (last 30 days)
            if 'hourly_recent' in data:
                recent = data['hourly_recent'][data['hourly_recent']['station'] == selected_station]
                if not recent.empty and 'pm25_ugm3' in recent.columns:
                    fig.add_trace(go.Scatter(
                        x=recent['datetime'], y=recent['pm25_ugm3'],
                        mode='lines', name='Datos Recientes',
                        line=dict(color='#2196F3', width=1),
                        opacity=0.6
                    ))

            # Hourly forecast
            if not fcast_h_st.empty:
                pred_col = [c for c in fcast_h_st.columns if 'predicted' in c.lower() or 'xgb' in c.lower()]
                if pred_col:
                    fig.add_trace(go.Scatter(
                        x=fcast_h_st['ds'], y=fcast_h_st[pred_col[0]],
                        mode='lines', name='Pronóstico 10 días',
                        line=dict(color='#FF9800', width=2.5),
                        fill='tozeroy', fillcolor='rgba(255, 152, 0, 0.1)'
                    ))

            fig.add_hline(y=NECAA_24H, line_dash="dot", line_color="red",
                         annotation_text="NECAA 24h (50 µg/m³)")

            fig.update_layout(
                title=f"PM2.5 Pronóstico Horario — {selected_station} (próximos 10 días)",
                xaxis_title="Fecha/Hora",
                yaxis_title="PM2.5 (µg/m³)",
                height=500,
                template="plotly_white",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Datos de pronóstico horario no disponibles. Ejecute el notebook primero.")

# ============================================================
# TAB 2: HISTORICAL ANALYSIS
# ============================================================
with tab2:
    st.markdown("### 📊 Análisis Histórico de PM2.5")

    if 'monthly' in data and 'pm25_ugm3' in data['monthly'].columns:
        df_m = data['monthly']

        # Time series for all stations
        fig_all = px.line(
            df_m, x='datetime', y='pm25_ugm3', color='station',
            title='PM2.5 Mensual — Todas las Estaciones (2004–2025)',
            labels={'pm25_ugm3': 'PM2.5 (µg/m³)', 'datetime': 'Fecha', 'station': 'Estación'},
            template='plotly_white'
        )
        fig_all.add_hline(y=WHO_GUIDELINE, line_dash="dot", line_color="red",
                         annotation_text="OMS")
        fig_all.update_layout(height=450)
        st.plotly_chart(fig_all, use_container_width=True)

        # Seasonality analysis
        col1, col2 = st.columns(2)

        with col1:
            # Monthly boxplot
            df_m['month_name'] = df_m['datetime'].dt.month_name()
            df_m['month_num'] = df_m['datetime'].dt.month
            fig_season = px.box(
                df_m.sort_values('month_num'), x='month_name', y='pm25_ugm3',
                color='month_name',
                title='Estacionalidad Mensual de PM2.5',
                labels={'pm25_ugm3': 'PM2.5 (µg/m³)', 'month_name': 'Mes'},
                template='plotly_white'
            )
            fig_season.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_season, use_container_width=True)

        with col2:
            # Annual trend
            annual = df_m.groupby(df_m['datetime'].dt.year)['pm25_ugm3'].mean().reset_index()
            annual.columns = ['year', 'pm25_avg']
            fig_annual = px.bar(
                annual, x='year', y='pm25_avg',
                title='Tendencia Anual PM2.5 (promedio todas las estaciones)',
                labels={'pm25_avg': 'PM2.5 Promedio (µg/m³)', 'year': 'Año'},
                template='plotly_white',
                color='pm25_avg',
                color_continuous_scale='YlOrRd'
            )
            fig_annual.add_hline(y=WHO_GUIDELINE, line_dash="dot", line_color="red")
            fig_annual.update_layout(height=400)
            st.plotly_chart(fig_annual, use_container_width=True)

        # Station comparison
        st.markdown("#### Comparación entre Estaciones")
        station_avg = df_m.groupby('station')['pm25_ugm3'].agg(['mean', 'std', 'max']).round(2)
        station_avg.columns = ['Media (µg/m³)', 'Desv. Est.', 'Máximo (µg/m³)']
        station_avg = station_avg.sort_values('Media (µg/m³)', ascending=False)
        st.dataframe(station_avg, use_container_width=True)
    else:
        st.warning("⚠️ Datos históricos mensuales no disponibles.")

# ============================================================
# TAB 3: MODEL COMPARISON
# ============================================================
with tab3:
    st.markdown("### 🏆 Comparación de Modelos")

    if 'comparison' in data:
        comp = data['comparison']

        # Display metrics table
        st.dataframe(comp.style.highlight_min(
            subset=['RMSE (avg)', 'MAE (avg)', 'MAPE (avg) %'],
            color='#90EE90'
        ), use_container_width=True)

        # Bar chart comparison
        fig_comp = make_subplots(rows=1, cols=3,
                                 subplot_titles=['RMSE', 'MAE', 'MAPE (%)'])

        colors = px.colors.qualitative.Set2[:len(comp)]
        for i, metric in enumerate(['RMSE (avg)', 'MAE (avg)', 'MAPE (avg) %']):
            if metric in comp.columns:
                fig_comp.add_trace(
                    go.Bar(x=comp['Model'], y=comp[metric], name=metric,
                           marker_color=colors, showlegend=False),
                    row=1, col=i+1
                )

        fig_comp.update_layout(
            title="Comparación de Métricas por Modelo",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Best model callout
        best = comp.sort_values('RMSE (avg)').iloc[0]
        st.success(f"🏆 **Mejor Modelo: {best['Model']}** — RMSE: {best['RMSE (avg)']:.2f} µg/m³")
    else:
        st.info("📊 Ejecute el notebook para generar la comparación de modelos.")

    # Example output table (as shown in project requirements)
    st.markdown("#### 📋 Ejemplo de Resultados")
    example_data = pd.DataFrame({
        'Station': ['Belisario', 'Tumbaco', 'El Camal', 'Carapungo'],
        'Horizon': ['5 years', '10 days (hourly)', '5 years', '10 days (hourly)'],
        'RMSE (Monthly)': ['5.2 µg/m³', '3.8 µg/m³', '6.1 µg/m³', '4.2 µg/m³'],
        'Forecast (2025)': ['45 µg/m³', '38 µg/m³ (Day 5)', '52 µg/m³', '35 µg/m³ (Day 5)']
    })
    st.table(example_data)

# ============================================================
# TAB 4: MAP
# ============================================================
with tab4:
    st.markdown("### 🗺️ Estaciones de Monitoreo REMMAQ")

    # Create map data
    map_data = []
    for station, (lat, lon, elev) in STATION_COORDS.items():
        avg_pm25 = np.nan
        if 'monthly' in data and 'pm25_ugm3' in data['monthly'].columns:
            st_data = data['monthly'][data['monthly']['station'] == station]
            if not st_data.empty:
                recent = st_data[st_data['datetime'] >= st_data['datetime'].max() - pd.DateOffset(years=1)]
                avg_pm25 = recent['pm25_ugm3'].mean()

        map_data.append({
            'station': station,
            'lat': lat, 'lon': lon,
            'elevation': elev,
            'pm25_avg_last_year': avg_pm25
        })

    df_map = pd.DataFrame(map_data)

    if not df_map['pm25_avg_last_year'].isna().all():
        fig_map = px.scatter_mapbox(
            df_map, lat='lat', lon='lon',
            color='pm25_avg_last_year',
            size='elevation',
            hover_name='station',
            hover_data={'elevation': True, 'pm25_avg_last_year': ':.1f'},
            color_continuous_scale='YlOrRd',
            size_max=20,
            zoom=10,
            title='Estaciones REMMAQ — PM2.5 Promedio Último Año'
        )
        fig_map.update_layout(
            mapbox_style="carto-positron",
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.map(df_map[['lat', 'lon']].rename(columns={'lat': 'latitude', 'lon': 'longitude'}))

    st.dataframe(df_map.round(2), use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    📊 Forecasting PM2.5 Air Pollution in Quito | REMMAQ 2004–2025<br>
    Diego Toscano & Andrés Guamán — Universidad de las Américas (UDLA)<br>
    Inteligencia Artificial II — Proyecto Final (Progreso 3) | Febrero 2026
</div>
""", unsafe_allow_html=True)
