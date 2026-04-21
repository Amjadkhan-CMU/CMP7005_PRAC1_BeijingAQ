# app.py — CMP7005 PRAC1 | Beijing Air Quality Dashboard
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import calendar

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beijing Air Quality — CMP7005",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    .inference-box {
        background: #EBF5FB;
        border-left: 5px solid #2E86C1;
        padding: 14px 18px;
        border-radius: 4px;
        margin-top: 0.8rem;
        font-size: 0.92rem;
        color: #1A252F;
    }

    div[data-testid="stSidebarContent"] { background: #1A252F; }
    div[data-testid="stSidebarContent"] * { color: #ECF0F1 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
STUDENT_ID   = "st20341331"
POLLUTANTS   = ['pm2.5', 'pm10', 'so2', 'no2', 'co', 'o3']
METEO        = ['temp', 'pres', 'dewp', 'rain', 'wspm', 'rh']
SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Autumn']
ZONE_COLORS  = {'Urban': '#C0392B', 'Suburban': '#2980B9'}
AQI_COLORS   = {
    'Good':          '#27AE60',
    'Moderate':      '#F39C12',
    'Unhealthy':     '#E67E22',
    'Very Unhealthy':'#E74C3C',
    'Hazardous':     '#8E44AD'
}

# ── Helper: coloured metric card ───────────────────────────────────────────────
def metric_card(label, value, bg_color):
    st.markdown(f"""
    <div style="background:{bg_color}; color:#FDFEFE; padding:18px 20px;
                border-radius:10px; text-align:center; margin-bottom:10px;">
        <p style="margin:0 0 6px 0; font-size:0.88rem; font-weight:600;
                  opacity:0.85;">{label}</p>
        <p style="margin:0; font-size:2rem; font-weight:800;">{value}</p>
    </div>""", unsafe_allow_html=True)

# ── Helper: inference box ──────────────────────────────────────────────────────
def inference(text):
    st.markdown(f'<div class="inference-box">💡 {text}</div>',
                unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    path = "data/beijing_aq_clean.csv"
    if not os.path.exists(path):
        st.error(f"Data file not found at '{path}'. Please run the notebook first.")
        st.stop()
    return pd.read_csv(path, index_col=0, parse_dates=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    path = "models/rf_pm25_model.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

df    = load_data()
model = load_model()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌏 Beijing AQ Dashboard")
    st.markdown("---")
    page = st.radio("Navigate to:", [
        "🏠  Home",
        "📊  Dataset Explorer",
        "📈  Visualisations",
        "🤖  PM2.5 Predictor",
        "📋  Model Report"
    ])
    st.markdown("---")
    st.markdown(f"""
**Module:** CMP7005 \n
**Module Leader:**  Dr. Amrita Prasad \n
**Student ID:** {STUDENT_ID} \n
**Student Name:** Amjad Hossain Khan**
""")


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":

    st.title("🌏 Beijing Air Quality Analysis")
    st.markdown(f"**CMP7005 PRAC1 — From Data to Application Development | {STUDENT_ID}**")
    st.markdown("---")

    # 4 metric cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("📁 Total Records", f"{len(df):,}", "#2C3E50")
    with c2:
        metric_card("📍 Monitoring Stations", df['station'].nunique(), "#154360")
    with c3:
        urban_mean = df[df['zone_type'] == 'Urban']['pm2.5'].mean()
        metric_card("🔴 Urban Mean PM2.5", f"{urban_mean:.1f} µg/m³", "#922B21")
    with c4:
        suburban_mean = df[df['zone_type'] == 'Suburban']['pm2.5'].mean()
        metric_card("🔵 Suburban Mean PM2.5", f"{suburban_mean:.1f} µg/m³", "#1A5276")

    st.markdown("---")

    # About + stations table
    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.subheader("About This Project")
        st.markdown("""
This dashboard presents the analysis of hourly air quality data from
**four Beijing monitoring stations** across **March 2013 to February 2017**.

The full pipeline:
- Task 1 — Data ingestion and merging (4 CSVs, 140,256 rows)
- Task 2 — Exploratory Data Analysis with 25 visualisations
- Task 3 — Random Forest PM2.5 prediction model
- Task 4 — This Streamlit application
- Task 5 — GitHub version control with 33 commits
        """)

    with col_r:
        st.subheader("Selected Stations")
        stations_df = pd.DataFrame({
            "Station":  ["Nongzhanguan", "Wanshouxigong", "Shunyi", "Dingling"],
            "Zone":     ["Urban", "Urban", "Suburban", "Suburban"],
            "District": ["Chaoyang", "Fengtai", "Shunyi", "Changping"],
            "Mean PM2.5": [
                f"{df[df['station'] == s]['pm2.5'].mean():.1f} µg/m³"
                for s in ["Nongzhanguan", "Wanshouxigong", "Shunyi", "Dingling"]
            ]
        })
        st.dataframe(stations_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Overview chart
    st.subheader("Monthly Average PM2.5 — All Stations")
    monthly = (df.groupby(["station", pd.Grouper(freq="ME")])["pm2.5"]
               .mean().reset_index())
    monthly.columns = ["station", "date", "pm2.5"]

    fig = px.line(
        monthly, x="date", y="pm2.5", color="station",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"pm2.5": "PM2.5 (µg/m³)", "date": ""}
    )
    fig.add_hline(y=75, line_dash="dash", line_color="orange",
                  annotation_text="China Grade II (75 µg/m³)")
    fig.add_hline(y=15, line_dash="dot", line_color="green",
                  annotation_text="WHO Annual Guideline (15 µg/m³)")
    fig.update_layout(template="simple_white", height=380, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    inference("<b>Key Observation:</b> All four stations exceed China's Grade II standard "
              "(75 µg/m³) during winter months. Even the cleanest suburban station (Dingling) "
              "consistently exceeds WHO guidelines — showing that Beijing's air quality "
              "problem is regional, not just a city-centre issue.")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dataset Explorer":

    st.title("📊 Dataset Explorer")
    st.markdown("Browse, filter and summarise the cleaned Beijing air quality dataset.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    sel_stations = c1.multiselect(
        "Station(s)", df["station"].unique().tolist(),
        default=df["station"].unique().tolist()
    )
    sel_years = c2.multiselect(
        "Year(s)", sorted(df["year"].unique().tolist()),
        default=sorted(df["year"].unique().tolist())
    )
    sel_zones = c3.multiselect(
        "Zone Type", ["Urban", "Suburban"],
        default=["Urban", "Suburban"]
    )

    filtered = df[
        df["station"].isin(sel_stations) &
        df["year"].isin(sel_years) &
        df["zone_type"].isin(sel_zones)
    ]

    st.markdown(f"**Showing {len(filtered):,} rows** (of {len(df):,} total)")

    tab1, tab2, tab3 = st.tabs([
        "📄 Data Table", "📊 Summary Statistics", "❓ Missing Values"
    ])

    with tab1:
        st.dataframe(filtered.head(2000), use_container_width=True)
        st.download_button(
            "⬇️ Download Filtered CSV",
            filtered.to_csv().encode("utf-8"),
            "beijing_filtered.csv",
            "text/csv"
        )

    with tab2:
        st.dataframe(
            filtered[POLLUTANTS + METEO].describe().round(2),
            use_container_width=True
        )

    with tab3:
        mv = filtered[POLLUTANTS + METEO].isnull().sum().reset_index()
        mv.columns = ["Column", "Missing Count"]
        mv["Missing %"] = (
            mv["Missing Count"] / max(len(filtered), 1) * 100
        ).round(2)
        st.dataframe(mv, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Visualisations":

    st.title("📈 Visualisations")
    st.markdown("Explore Beijing air quality patterns interactively.")
    st.markdown("---")

    viz = st.selectbox("Choose a Visualisation:", [
        "Time Series — Monthly Average by Station",
        "Distribution — Pollutant Histogram",
        "Seasonal Patterns",
        "Diurnal (Hourly) Pattern",
        "Urban vs Suburban Box Plot",
        "Correlation Heatmap",
        "AQI Category Distribution",
        "PM2.5 Monthly Heatmap",
    ])
    st.markdown("---")

    # Time Series
    if viz == "Time Series — Monthly Average by Station":
        pol = st.selectbox("Pollutant", POLLUTANTS, index=0)

        monthly = (df.groupby(["station", pd.Grouper(freq="ME")])[pol]
                   .mean().reset_index())
        monthly.columns = ["station", "date", pol]

        fig = px.line(
            monthly, x="date", y=pol, color="station",
            title=f"Monthly Average {pol.upper()} by Station (2013-2017)",
            labels={pol: f"{pol.upper()} (µg/m³)"},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        if pol == "pm2.5":
            fig.add_hline(y=75, line_dash="dash", line_color="orange",
                          annotation_text="China Grade II (75)")
            fig.add_hline(y=15, line_dash="dot", line_color="green",
                          annotation_text="WHO (15)")
        fig.update_layout(template="simple_white", height=460)
        st.plotly_chart(fig, use_container_width=True)
        inference("<b>Inference:</b> Winter months (Dec-Feb) consistently show the highest "
                  "concentrations. Urban stations are always higher than suburban, "
                  "and the gap grows in winter due to coal heating.")

    # Histogram
    elif viz == "Distribution — Pollutant Histogram":
        pol   = st.selectbox("Pollutant", POLLUTANTS)
        split = st.checkbox("Split by Zone Type", value=True)

        fig = px.histogram(
            df, x=pol,
            color="zone_type" if split else None,
            nbins=80, barmode="overlay", opacity=0.72,
            title=f"{pol.upper()} Distribution",
            color_discrete_map=ZONE_COLORS,
            labels={pol: f"{pol.upper()} (µg/m³)"}
        )
        fig.update_layout(template="simple_white", height=440)
        st.plotly_chart(fig, use_container_width=True)

        skew   = df[pol].skew()
        mean_v = df[pol].mean()
        med_v  = df[pol].median()
        inference(f"<b>Inference:</b> {pol.upper()} has a skewness of <b>{skew:.2f}</b>. "
                  f"Most hours have low values but occasional spikes pull the mean "
                  f"({mean_v:.0f}) above the median ({med_v:.0f} µg/m³).")

    # Seasonal
    elif viz == "Seasonal Patterns":
        pol   = st.selectbox("Pollutant", POLLUTANTS)
        chart = st.radio("Chart Type",
                         ["Grouped Bar", "Box Plot", "Heatmap"],
                         horizontal=True)

        if chart == "Grouped Bar":
            grp = df.groupby(["season", "zone_type"])[pol].mean().reset_index()
            grp["season"] = pd.Categorical(grp["season"], SEASON_ORDER)
            grp.sort_values("season", inplace=True)
            fig = px.bar(grp, x="season", y=pol, color="zone_type",
                         barmode="group",
                         title=f"Average {pol.upper()} by Season and Zone",
                         color_discrete_map=ZONE_COLORS,
                         labels={pol: f"Mean {pol.upper()} (µg/m³)"})
            fig.update_layout(template="simple_white", height=440)
            st.plotly_chart(fig, use_container_width=True)

        elif chart == "Box Plot":
            data = df.copy()
            data["season"] = pd.Categorical(data["season"], SEASON_ORDER)
            fig = px.box(data, x="season", y=pol, color="zone_type",
                         points=False, title=f"{pol.upper()} by Season",
                         color_discrete_map=ZONE_COLORS)
            fig.update_layout(template="simple_white", height=460)
            st.plotly_chart(fig, use_container_width=True)

        else:
            pivot = df.groupby(["zone_type", "season"])[pol].mean().unstack()
            pivot = pivot[SEASON_ORDER]
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                        linewidths=0.4, ax=ax,
                        cbar_kws={"label": f"{pol.upper()} µg/m³"})
            ax.set_title(f"Mean {pol.upper()} by Zone and Season",
                         fontweight="bold")
            st.pyplot(fig)
            plt.close()

        sm    = df.groupby("season")[pol].mean().reindex(SEASON_ORDER)
        top_s = sm.idxmax()
        low_s = sm.idxmin()
        inference(f"<b>Inference:</b> {pol.upper()} peaks in <b>{top_s}</b> "
                  f"({sm[top_s]:.1f} µg/m³) and is lowest in <b>{low_s}</b> "
                  f"({sm[low_s]:.1f} µg/m³).")

    # Diurnal
    elif viz == "Diurnal (Hourly) Pattern":
        pol   = st.selectbox("Pollutant", POLLUTANTS)
        split = st.radio("Split by",
                         ["zone_type", "station", "season"],
                         horizontal=True)
        hourly = df.groupby(["hour", split])[pol].mean().reset_index()
        fig = px.line(
            hourly, x="hour", y=pol, color=split,
            title=f"Diurnal {pol.upper()} Pattern",
            labels={pol: f"{pol.upper()} (µg/m³)", "hour": "Hour of Day"},
            color_discrete_map=ZONE_COLORS if split == "zone_type" else None,
            markers=True
        )
        fig.update_xaxes(tickvals=list(range(0, 24, 2)))
        fig.update_layout(template="simple_white", height=460)
        st.plotly_chart(fig, use_container_width=True)
        inference("<b>Inference:</b> PM2.5 and NO2 show a double-peak pattern — "
                  "morning rush hour (~08:00) and evening (~20:00). "
                  "O3 peaks ~14:00 from photochemical reactions.")

    # Box Plot
    elif viz == "Urban vs Suburban Box Plot":
        pol = st.selectbox("Pollutant", POLLUTANTS)
        fig = px.box(
            df, x="station", y=pol, color="zone_type",
            points=False,
            title=f"{pol.upper()} Distribution by Station",
            color_discrete_map=ZONE_COLORS,
            category_orders={"station": ["Nongzhanguan", "Wanshouxigong",
                                         "Shunyi", "Dingling"]}
        )
        fig.update_layout(template="simple_white", height=460)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    elif viz == "Correlation Heatmap":
        station_sel = st.selectbox("Station (or All)",
                                   ["All"] + df["station"].unique().tolist())
        cols = POLLUTANTS + ["temp", "pres", "wspm", "rh"]
        sub  = df if station_sel == "All" else df[df["station"] == station_sel]
        corr = sub[cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    linewidths=0.4, ax=ax, annot_kws={"size": 9}, square=True)
        ax.set_title(f"Correlation Matrix — {station_sel}",
                     fontweight="bold", fontsize=12)
        st.pyplot(fig)
        plt.close()
        inference("<b>Key correlations:</b> PM2.5 and PM10 (r = +0.86) share combustion "
                  "sources. PM2.5 and Temperature (r = -0.45) — cold air traps pollutants. "
                  "O3 and NO2 (r = -0.36) — urban ozone titration effect.")

    # AQI Category
    elif viz == "AQI Category Distribution":
        group_by  = st.radio("Group by",
                             ["All", "zone_type", "season", "station"],
                             horizontal=True)
        aqi_order = list(AQI_COLORS.keys())

        if group_by == "All":
            counts = (df["aqi_category"].value_counts()
                      .reindex(aqi_order).reset_index())
            counts.columns = ["aqi_category", "count"]
            fig = px.bar(counts, x="aqi_category", y="count",
                         color="aqi_category", color_discrete_map=AQI_COLORS,
                         title="AQI Category Distribution",
                         labels={"count": "Number of Hours",
                                 "aqi_category": "AQI Category"})
        else:
            counts = (df.groupby([group_by, "aqi_category"])
                      .size().reset_index(name="count").dropna())
            counts["aqi_category"] = pd.Categorical(counts["aqi_category"],
                                                     aqi_order)
            counts.sort_values("aqi_category", inplace=True)
            fig = px.bar(counts, x=group_by, y="count",
                         color="aqi_category", color_discrete_map=AQI_COLORS,
                         barmode="stack",
                         title=f"AQI Category by {group_by}")

        fig.update_layout(template="simple_white", height=460)
        st.plotly_chart(fig, use_container_width=True)

        pct      = (df["aqi_category"].value_counts(normalize=True) * 100).round(1)
        good_pct = pct.get("Good", 0)
        bad_pct  = pct.get("Hazardous", 0) + pct.get("Very Unhealthy", 0)
        inference(f"<b>Inference:</b> Only <b>{good_pct:.1f}%</b> of hourly readings are "
                  f"Good (PM2.5 below 35 µg/m³). <b>{bad_pct:.1f}%</b> of hours are "
                  f"Very Unhealthy or Hazardous.")

    # Monthly Heatmap
    elif viz == "PM2.5 Monthly Heatmap":
        station_sel = st.selectbox("Station",
                                   ["All"] + df["station"].unique().tolist())
        sub   = df if station_sel == "All" else df[df["station"] == station_sel]
        pivot = sub.groupby(["year", "month"])["pm2.5"].mean().unstack()
        pivot.index   = pivot.index.astype(str)
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig, ax = plt.subplots(figsize=(14, 4))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                    linewidths=0.3, cbar_kws={"label": "PM2.5 (µg/m³)"}, ax=ax)
        ax.set_title(f"Monthly Average PM2.5 — {station_sel}",
                     fontweight="bold", fontsize=12)
        ax.set_ylabel("Year")
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PM2.5 PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  PM2.5 Predictor":

    st.title("🤖 PM2.5 Concentration Predictor")
    st.markdown("Enter values below — the trained **Random Forest** model will "
                "predict the hourly PM2.5 concentration.")
    st.markdown("---")

    if model is None:
        st.error("Model not found at `models/rf_pm25_model.pkl`. "
                 "Please run the notebook first.")
        st.stop()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Other Pollutants")
        pm10 = st.number_input("PM10 (µg/m³)",   0.0, 800.0,   80.0,  step=5.0)
        so2  = st.number_input("SO2 (µg/m³)",    0.0, 400.0,   20.0,  step=2.0)
        no2  = st.number_input("NO2 (µg/m³)",    0.0, 250.0,   50.0,  step=2.0)
        co   = st.number_input("CO (µg/m³)",     0.0, 12000.0, 800.0, step=50.0)
        o3   = st.number_input("O3 (µg/m³)",     0.0, 350.0,   60.0,  step=5.0)

    with c2:
        st.subheader("Meteorological")
        temp = st.number_input("Temperature (°C)", -25.0, 45.0,   15.0,   step=1.0)
        pres = st.number_input("Pressure (hPa)",   985.0, 1050.0, 1013.0, step=1.0)
        dewp = st.number_input("Dew Point (°C)",  -45.0,  35.0,   5.0,   step=1.0)
        rain = st.number_input("Rainfall (mm)",     0.0,  100.0,  0.0,   step=0.5)
        wspm = st.number_input("Wind Speed (m/s)",  0.0,   20.0,  2.0,   step=0.5)
        rh   = st.number_input("Rel. Humidity (%)", 0.0,  100.0, 55.0,   step=1.0)

    with c3:
        st.subheader("Time & Location")
        hour    = st.selectbox("Hour of Day", list(range(24)), index=8)
        month   = st.selectbox("Month", list(range(1, 13)),
                               format_func=lambda m: calendar.month_name[m])
        day_str = st.selectbox("Day of Week",
                               ["Monday", "Tuesday", "Wednesday", "Thursday",
                                "Friday", "Saturday", "Sunday"])
        dow     = ["Monday", "Tuesday", "Wednesday", "Thursday",
                   "Friday", "Saturday", "Sunday"].index(day_str)
        weekend  = 1 if dow >= 5 else 0
        zone     = st.radio("Zone Type", ["Urban", "Suburban"])
        zone_enc = 1 if zone == "Urban" else 0

    st.markdown("---")

    if st.button("🔮 Predict PM2.5 Concentration", type="primary",
                 use_container_width=True):

        input_row = pd.DataFrame([[
            pm10, so2, no2, co, o3,
            temp, pres, dewp, rain, wspm, rh,
            hour, month, dow, weekend, zone_enc
        ]], columns=[
            "pm10", "so2", "no2", "co", "o3",
            "temp", "pres", "dewp", "rain", "wspm", "rh",
            "hour", "month", "dayofweek", "is_weekend", "zone_type"
        ])

        prediction = float(max(0.0, model.predict(input_row)[0]))

        if   prediction <= 35:  cat, emoji = "Good",           "🟢"
        elif prediction <= 75:  cat, emoji = "Moderate",       "🟡"
        elif prediction <= 115: cat, emoji = "Unhealthy",      "🟠"
        elif prediction <= 150: cat, emoji = "Very Unhealthy", "🔴"
        else:                   cat, emoji = "Hazardous",      "🟣"

        col_res, col_gauge = st.columns([1, 1.6])

        with col_res:
            st.metric("🎯 Predicted PM2.5", f"{prediction:.1f} µg/m³")
            st.markdown(f"### {emoji} &nbsp; **{cat}**")
            st.markdown("*(China GB3095-2012 Standard)*")
            st.markdown("---")
            st.markdown("""
| Reference | Limit |
|-----------|-------|
| China Grade II (annual)  | 35 µg/m³ |
| China Grade II (24-hour) | 75 µg/m³ |
| WHO Annual Guideline     | 5 µg/m³  |
""")

        with col_gauge:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={"text": "PM2.5 (µg/m³)", "font": {"size": 15}},
                number={"suffix": " µg/m³", "font": {"size": 22}},
                gauge={
                    "axis": {"range": [0, 250], "tickwidth": 1},
                    "steps": [
                        {"range": [0,   35],  "color": "#27AE60"},
                        {"range": [35,  75],  "color": "#F39C12"},
                        {"range": [75,  115], "color": "#E67E22"},
                        {"range": [115, 150], "color": "#E74C3C"},
                        {"range": [150, 250], "color": "#8E44AD"},
                    ],
                    "bar":       {"color": "#2C3E50", "thickness": 0.25},
                    "threshold": {"line":      {"color": "red", "width": 3},
                                  "thickness": 0.75, "value": 75}
                }
            ))
            fig_g.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig_g, use_container_width=True)

        exceeds = "Exceeds" if prediction > 75 else "Within"
        inference(f"<b>Result:</b> Predicted <b>{prediction:.1f} µg/m³</b> PM2.5 — "
                  f"classified as <b>{cat}</b> (China GB3095-2012). "
                  f"{exceeds} the China 24-hour Grade II limit of 75 µg/m³.")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋  Model Report":

    st.title("📋 Model Performance Report")
    st.markdown("---")

    perf_path = "models/model_performance.csv"
    if os.path.exists(perf_path):
        st.subheader("Model Comparison")
        st.dataframe(pd.read_csv(perf_path), hide_index=True,
                     use_container_width=True)
    else:
        st.info("Run the notebook to generate model_performance.csv")

    st.markdown("---")

    st.subheader("Model Configuration")
    st.markdown("""
| Parameter | Value |
|-----------|-------|
| **Algorithm** | Random Forest Regressor |
| **Target** | PM2.5 (µg/m³) |
| **Train / Test Split** | 80% / 20% — chronological |
| **Tuning** | GridSearchCV, 3-fold CV |
| **Metrics** | MAE, RMSE, R² |

**Input Features (16 total):** PM10, SO2, NO2, CO, O3, Temperature,
Pressure, Dew Point, Rainfall, Wind Speed, Relative Humidity,
Hour, Month, Day of Week, Weekend flag, Zone Type
""")

    st.markdown("---")

    st.subheader("Saved Figures from Notebook")
    fig_dir = "outputs"
    if os.path.exists(fig_dir):
        figs = sorted([f for f in os.listdir(fig_dir) if f.endswith(".png")])
        if figs:
            cols = st.columns(2)
            for i, ff in enumerate(figs):
                cols[i % 2].image(
                    os.path.join(fig_dir, ff),
                    caption=ff.replace("_", " ").replace(".png", ""),
                    use_container_width=True
                )
        else:
            st.info("No PNG figures found. Run the notebook first.")
    else:
        st.info("outputs/ folder not found. Run the notebook first.")
