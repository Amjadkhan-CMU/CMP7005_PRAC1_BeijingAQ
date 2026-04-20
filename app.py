# app.py — CMP7005 PRAC1 | Beijing Air Quality Interactive Dashboard
# Cardiff Metropolitan University | School of Technologies | 2025–26
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle, os, calendar

st.set_page_config(
    page_title="Beijing Air Quality — CMP7005",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { font-size: 2rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.4rem !important; font-weight: 600 !important; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 0.4rem; }
    .inference-box {
        background: #EBF5FB; border-left: 5px solid #2E86C1;
        padding: 14px 18px; border-radius: 4px;
        margin-top: 0.8rem; font-size: 0.92rem;
    }
    div[data-testid="stSidebarContent"] { background: #1A252F; }
    div[data-testid="stSidebarContent"] * { color: #ECF0F1 !important; }
</style>
""", unsafe_allow_html=True)

POLLUTANTS   = ['pm2.5','pm10','so2','no2','co','o3']
METEO        = ['temp','pres','dewp','rain','wspm','rh']
SEASON_ORDER = ['Winter','Spring','Summer','Autumn']
ZONE_COLORS  = {'Urban':'#C0392B','Suburban':'#2980B9'}
AQI_COLORS   = {'Good':'#27AE60','Moderate':'#F39C12','Unhealthy':'#E67E22',
                'Very Unhealthy':'#E74C3C','Hazardous':'#8E44AD'}

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    path = "data/beijing_aq_clean.csv"
    if not os.path.exists(path):
        st.error(f"Data file not found at '{path}'. Run the main notebook first.")
        st.stop()
    return pd.read_csv(path, index_col=0, parse_dates=True)

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = "models/rf_pm25_model.pkl"
    if not os.path.exists(path):
        return None
    with open(path,"rb") as f:
        return pickle.load(f)

df    = load_data()
model = load_model()

with st.sidebar:
    st.markdown("## 🌏 Beijing AQ Dashboard")
    st.markdown("---")
    page = st.radio("Navigate to:", [
        "🏠  Home","📊  Dataset Explorer","📈  Visualisations",
        "🤖  PM2.5 Predictor","📋  Model Report"
    ])
    st.markdown("---")
    st.markdown("**Module:** CMP7005 PRAC1  \n**Cardiff Met** | 2025–26  \n**Dataset:** PRSA Beijing 2013–2017")

# ── HOME ──────────────────────────────────────────────────────────────────────
if page == "🏠  Home":
    st.title("🌏 Beijing Air Quality Analysis")
    st.markdown("**CMP7005 PRAC1 — From Data to Application Development | Cardiff Metropolitan University**")
    st.markdown("---")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📁 Total Records",      f"{len(df):,}")
    c2.metric("📍 Stations",           str(df['station'].nunique()))
    c3.metric("🔴 Urban Mean PM2.5",   f"{df[df['zone_type']=='Urban']['pm2.5'].mean():.1f} µg/m³")
    c4.metric("🔵 Suburban Mean PM2.5",f"{df[df['zone_type']=='Suburban']['pm2.5'].mean():.1f} µg/m³")

    st.markdown("---")
    col_l, col_r = st.columns([1.2,1])
    with col_l:
        st.subheader("About This Project")
        st.markdown("""
This dashboard is the interactive component of **CMP7005 PRAC1**, analysing hourly air quality
data from four Beijing monitoring stations across **March 2013 – February 2017**.

The full pipeline:
- ✅ **Task 1** — Data ingestion and merging (4 CSVs → 140,256 rows)
- ✅ **Task 2** — Exploratory Data Analysis with 25 visualisations
- ✅ **Task 3** — Random Forest PM2.5 prediction model (R² > 0.90)
- ✅ **Task 4** — This Streamlit application
- ✅ **Task 5** — GitHub version control with descriptive commits
        """)
    with col_r:
        st.subheader("Selected Stations")
        st.dataframe(pd.DataFrame({
            "Station": ["Nongzhanguan","Wanshouxigong","Shunyi","Dingling"],
            "Zone":    ["🔴 Urban","🔴 Urban","🔵 Suburban","🔵 Suburban"],
            "District":["Chaoyang","Fengtai","Shunyi","Changping"],
            "Mean PM2.5":[
                f"{df[df['station']=='Nongzhanguan']['pm2.5'].mean():.1f} µg/m³",
                f"{df[df['station']=='Wanshouxigong']['pm2.5'].mean():.1f} µg/m³",
                f"{df[df['station']=='Shunyi']['pm2.5'].mean():.1f} µg/m³",
                f"{df[df['station']=='Dingling']['pm2.5'].mean():.1f} µg/m³",
            ]
        }), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Monthly Average PM2.5 — All Stations")
    monthly = df.groupby(["station",pd.Grouper(freq="ME")])["pm2.5"].mean().reset_index()
    monthly.columns = ["station","date","pm2.5"]
    fig = px.line(monthly, x="date", y="pm2.5", color="station",
                  color_discrete_sequence=px.colors.qualitative.Bold,
                  labels={"pm2.5":"PM2.5 (µg/m³)","date":""})
    fig.add_hline(y=75,line_dash="dash",line_color="orange",annotation_text="China Grade II (75 µg/m³)")
    fig.add_hline(y=15,line_dash="dot",line_color="green",annotation_text="WHO Annual Guideline (15 µg/m³)")
    fig.update_layout(template="simple_white",height=380,margin=dict(t=20))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="inference-box">💡 <b>Key Observation:</b> All four stations exceed China\'s Grade II standard (75 µg/m³) during winter months. Even the cleanest station (Dingling, suburban) consistently exceeds WHO guidelines — confirming that Beijing\'s air quality problem is regional in scale.</div>', unsafe_allow_html=True)

# ── DATASET EXPLORER ──────────────────────────────────────────────────────────
elif page == "📊  Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.markdown("Browse, filter and summarise the cleaned Beijing air quality dataset.")
    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    sel_stations = c1.multiselect("Station(s)", df["station"].unique().tolist(), default=df["station"].unique().tolist())
    sel_years    = c2.multiselect("Year(s)", sorted(df["year"].unique().tolist()), default=sorted(df["year"].unique().tolist()))
    sel_zones    = c3.multiselect("Zone Type", ["Urban","Suburban"], default=["Urban","Suburban"])
    filtered = df[df["station"].isin(sel_stations) & df["year"].isin(sel_years) & df["zone_type"].isin(sel_zones)]
    st.markdown(f"**Showing {len(filtered):,} rows** (of {len(df):,} total)")
    tab1,tab2,tab3 = st.tabs(["📄 Data Table","📊 Summary Statistics","❓ Missing Values"])
    with tab1:
        st.dataframe(filtered.head(2000), use_container_width=True)
        st.download_button("⬇️ Download Filtered CSV", filtered.to_csv().encode("utf-8"), "beijing_filtered.csv","text/csv")
    with tab2:
        st.dataframe(filtered[POLLUTANTS+METEO].describe().round(2), use_container_width=True)
    with tab3:
        mv = filtered[POLLUTANTS+METEO].isnull().sum().reset_index()
        mv.columns = ["Column","Missing Count"]
        mv["Missing %"] = (mv["Missing Count"]/max(len(filtered),1)*100).round(2)
        st.dataframe(mv, hide_index=True, use_container_width=True)

# ── VISUALISATIONS ────────────────────────────────────────────────────────────
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

    if viz == "Time Series — Monthly Average by Station":
        pol = st.selectbox("Pollutant", POLLUTANTS, index=0)
        monthly = df.groupby(["station",pd.Grouper(freq="ME")])[pol].mean().reset_index()
        monthly.columns = ["station","date",pol]
        fig = px.line(monthly,x="date",y=pol,color="station",
                      title=f"Monthly Average {pol.upper()} by Station (2013–2017)",
                      labels={pol:f"{pol.upper()} (µg/m³)"},
                      color_discrete_sequence=px.colors.qualitative.Bold)
        if pol=="pm2.5":
            fig.add_hline(y=75,line_dash="dash",line_color="orange",annotation_text="China Grade II (75)")
            fig.add_hline(y=15,line_dash="dot",line_color="green",annotation_text="WHO (15)")
        fig.update_layout(template="simple_white",height=460)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown('<div class="inference-box">💡 <b>Inference:</b> Winter months (Dec–Feb) consistently produce the highest concentrations. Urban stations track above suburban ones throughout the year, with the gap widening in winter when coal heating amplifies the urban emission premium.</div>',unsafe_allow_html=True)

    elif viz == "Distribution — Pollutant Histogram":
        pol   = st.selectbox("Pollutant", POLLUTANTS)
        split = st.checkbox("Split by Zone Type", value=True)
        fig = px.histogram(df,x=pol,color="zone_type" if split else None,nbins=80,
                           barmode="overlay",opacity=0.72,
                           title=f"{pol.upper()} Distribution",
                           color_discrete_map=ZONE_COLORS,
                           labels={pol:f"{pol.upper()} (µg/m³)"})
        fig.update_layout(template="simple_white",height=440)
        st.plotly_chart(fig,use_container_width=True)
        skew = df[pol].skew()
        st.markdown(f'<div class="inference-box">💡 <b>Inference:</b> {pol.upper()} has skewness of <b>{skew:.2f}</b> — strongly right-skewed, driven by episodic pollution events. Median ({df[pol].median():.0f} µg/m³) is more representative than mean ({df[pol].mean():.0f} µg/m³).</div>',unsafe_allow_html=True)

    elif viz == "Seasonal Patterns":
        pol   = st.selectbox("Pollutant", POLLUTANTS)
        chart = st.radio("Chart Type", ["Grouped Bar","Box Plot","Heatmap"], horizontal=True)
        if chart == "Grouped Bar":
            grp = df.groupby(["season","zone_type"])[pol].mean().reset_index()
            grp["season"] = pd.Categorical(grp["season"],SEASON_ORDER); grp.sort_values("season",inplace=True)
            fig = px.bar(grp,x="season",y=pol,color="zone_type",barmode="group",
                         title=f"Average {pol.upper()} by Season and Zone",
                         color_discrete_map=ZONE_COLORS,labels={pol:f"Mean {pol.upper()} (µg/m³)"})
            fig.update_layout(template="simple_white",height=440)
            st.plotly_chart(fig,use_container_width=True)
        elif chart == "Box Plot":
            data = df.copy(); data["season"] = pd.Categorical(data["season"],SEASON_ORDER)
            fig = px.box(data,x="season",y=pol,color="zone_type",points=False,
                         title=f"{pol.upper()} by Season",color_discrete_map=ZONE_COLORS)
            fig.update_layout(template="simple_white",height=460)
            st.plotly_chart(fig,use_container_width=True)
        else:
            pivot = df.groupby(["zone_type","season"])[pol].mean().unstack()[SEASON_ORDER]
            fig,ax = plt.subplots(figsize=(10,3))
            sns.heatmap(pivot,annot=True,fmt=".0f",cmap="YlOrRd",linewidths=0.4,ax=ax,cbar_kws={"label":f"{pol.upper()} µg/m³"})
            ax.set_title(f"Mean {pol.upper()} — Zone × Season",fontweight="bold")
            st.pyplot(fig); plt.close()
        sm = df.groupby("season")[pol].mean().reindex(SEASON_ORDER)
        st.markdown(f'<div class="inference-box">💡 <b>Inference:</b> {pol.upper()} peaks in <b>{sm.idxmax()}</b> ({sm.max():.1f} µg/m³) and is lowest in <b>{sm.idxmin()}</b> ({sm.min():.1f} µg/m³).</div>',unsafe_allow_html=True)

    elif viz == "Diurnal (Hourly) Pattern":
        pol   = st.selectbox("Pollutant", POLLUTANTS)
        split = st.radio("Split by", ["zone_type","station","season"], horizontal=True)
        hourly = df.groupby(["hour",split])[pol].mean().reset_index()
        fig = px.line(hourly,x="hour",y=pol,color=split,
                      title=f"Diurnal {pol.upper()} Pattern",
                      labels={pol:f"{pol.upper()} (µg/m³)","hour":"Hour of Day"},
                      color_discrete_map=ZONE_COLORS if split=="zone_type" else None,markers=True)
        fig.update_xaxes(tickvals=list(range(0,24,2)))
        fig.update_layout(template="simple_white",height=460)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown('<div class="inference-box">💡 <b>Inference:</b> PM2.5 and NO2 show a double-peak pattern: morning rush (~08:00) and evening (~20:00), separated by a midday dip. O3 peaks ~14:00 from photochemical reactions. Urban stations show sharper peaks than suburban.</div>',unsafe_allow_html=True)

    elif viz == "Urban vs Suburban Box Plot":
        pol = st.selectbox("Pollutant", POLLUTANTS)
        fig = px.box(df,x="station",y=pol,color="zone_type",points=False,
                     title=f"{pol.upper()} Distribution by Station",color_discrete_map=ZONE_COLORS,
                     category_orders={"station":["Nongzhanguan","Wanshouxigong","Shunyi","Dingling"]})
        fig.update_layout(template="simple_white",height=460)
        st.plotly_chart(fig,use_container_width=True)

    elif viz == "Correlation Heatmap":
        station_sel = st.selectbox("Station (or All)", ["All"]+df["station"].unique().tolist())
        cols = POLLUTANTS+["temp","pres","wspm","rh"]
        sub  = df if station_sel=="All" else df[df["station"]==station_sel]
        corr = sub[cols].corr()
        fig,ax = plt.subplots(figsize=(10,8))
        mask = np.triu(np.ones_like(corr,dtype=bool))
        sns.heatmap(corr,mask=mask,annot=True,fmt=".2f",cmap="RdBu_r",center=0,
                    vmin=-1,vmax=1,linewidths=0.4,ax=ax,annot_kws={"size":9},square=True)
        ax.set_title(f"Correlation Matrix — {station_sel}",fontweight="bold",fontsize=12)
        st.pyplot(fig); plt.close()
        st.markdown('<div class="inference-box">💡 <b>Key correlations:</b> PM2.5 ↔ PM10 (r≈+0.86) — shared combustion sources. PM2.5 ↔ Temperature (r≈−0.45) — winter inversions trap pollutants. O3 ↔ NO2 (r≈−0.36) — urban ozone titration. PM2.5 ↔ Wind Speed (r≈−0.30) — ventilation disperses particulates.</div>',unsafe_allow_html=True)

    elif viz == "AQI Category Distribution":
        group_by = st.radio("Group by", ["All","zone_type","season","station"], horizontal=True)
        aqi_order = list(AQI_COLORS.keys())
        if group_by=="All":
            counts = df["aqi_category"].value_counts().reindex(aqi_order).reset_index()
            counts.columns = ["aqi_category","count"]
            fig = px.bar(counts,x="aqi_category",y="count",color="aqi_category",color_discrete_map=AQI_COLORS,
                         title="AQI Category Distribution",labels={"count":"Number of Hours","aqi_category":"AQI Category"})
        else:
            counts = df.groupby([group_by,"aqi_category"]).size().reset_index(name="count").dropna()
            counts["aqi_category"] = pd.Categorical(counts["aqi_category"],aqi_order); counts.sort_values("aqi_category",inplace=True)
            fig = px.bar(counts,x=group_by,y="count",color="aqi_category",color_discrete_map=AQI_COLORS,barmode="stack",title=f"AQI Category by {group_by}")
        fig.update_layout(template="simple_white",height=460)
        st.plotly_chart(fig,use_container_width=True)
        pct = (df["aqi_category"].value_counts(normalize=True)*100).round(1)
        st.markdown(f'<div class="inference-box">💡 <b>Inference:</b> Only <b>{pct.get("Good",0):.1f}%</b> of hourly readings are Good (PM2.5 ≤ 35 µg/m³). <b>{pct.get("Hazardous",0)+pct.get("Very Unhealthy",0):.1f}%</b> of hours are Very Unhealthy or Hazardous — levels at which outdoor activity poses serious health risk.</div>',unsafe_allow_html=True)

    elif viz == "PM2.5 Monthly Heatmap":
        station_sel2 = st.selectbox("Station", ["All"]+df["station"].unique().tolist())
        sub = df if station_sel2=="All" else df[df["station"]==station_sel2]
        pivot = sub.groupby(["year","month"])["pm2.5"].mean().unstack()
        pivot.index = pivot.index.astype(str)
        pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig,ax = plt.subplots(figsize=(14,4))
        sns.heatmap(pivot,annot=True,fmt=".0f",cmap="YlOrRd",linewidths=0.3,cbar_kws={"label":"PM2.5 (µg/m³)"},ax=ax)
        ax.set_title(f"Monthly Average PM2.5 — {station_sel2}",fontweight="bold",fontsize=12); ax.set_ylabel("Year")
        st.pyplot(fig); plt.close()

# ── PM2.5 PREDICTOR ───────────────────────────────────────────────────────────
elif page == "🤖  PM2.5 Predictor":
    st.title("🤖 PM2.5 Concentration Predictor")
    st.markdown("Input values below and the trained **Random Forest** model predicts the hourly PM2.5 concentration.")
    st.markdown("---")
    if model is None:
        st.error("Model not found at `models/rf_pm25_model.pkl`. Run the notebook first."); st.stop()

    c1,c2,c3 = st.columns(3)
    with c1:
        st.subheader("Other Pollutants")
        pm10 = st.number_input("PM10 (µg/m³)",  0.0,800.0,80.0,step=5.0)
        so2  = st.number_input("SO2 (µg/m³)",   0.0,400.0,20.0,step=2.0)
        no2  = st.number_input("NO2 (µg/m³)",   0.0,250.0,50.0,step=2.0)
        co   = st.number_input("CO (µg/m³)",    0.0,12000.0,800.0,step=50.0)
        o3   = st.number_input("O3 (µg/m³)",    0.0,350.0,60.0,step=5.0)
    with c2:
        st.subheader("Meteorological")
        temp = st.number_input("Temperature (°C)",-25.0,45.0,15.0,step=1.0)
        pres = st.number_input("Pressure (hPa)",985.0,1050.0,1013.0,step=1.0)
        dewp = st.number_input("Dew Point (°C)",-45.0,35.0,5.0,step=1.0)
        rain = st.number_input("Rainfall (mm)",0.0,100.0,0.0,step=0.5)
        wspm = st.number_input("Wind Speed (m/s)",0.0,20.0,2.0,step=0.5)
        rh   = st.number_input("Rel. Humidity (%)",0.0,100.0,55.0,step=1.0)
    with c3:
        st.subheader("Time & Location")
        hour  = st.selectbox("Hour of Day",list(range(24)),index=8)
        month = st.selectbox("Month",list(range(1,13)),format_func=lambda m:calendar.month_name[m])
        dow_str = st.selectbox("Day of Week",["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        dow_num = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(dow_str)
        weekend = 1 if dow_num>=5 else 0
        zone    = st.radio("Zone Type",["Urban","Suburban"])
        zone_enc = 1 if zone=="Urban" else 0

    st.markdown("---")
    if st.button("🔮 Predict PM2.5 Concentration", type="primary", use_container_width=True):
        inp = pd.DataFrame([[pm10,so2,no2,co,o3,temp,pres,dewp,rain,wspm,rh,hour,month,dow_num,weekend,zone_enc]],
                           columns=["pm10","so2","no2","co","o3","temp","pres","dewp","rain","wspm","rh",
                                    "hour","month","dayofweek","is_weekend","zone_type"])
        prediction = float(max(0.0, model.predict(inp)[0]))
        if   prediction<=35:  cat,emoji="Good","🟢"
        elif prediction<=75:  cat,emoji="Moderate","🟡"
        elif prediction<=115: cat,emoji="Unhealthy","🟠"
        elif prediction<=150: cat,emoji="Very Unhealthy","🔴"
        else:                 cat,emoji="Hazardous","🟣"

        col_res,col_gauge = st.columns([1,1.6])
        with col_res:
            st.metric("🎯 Predicted PM2.5",f"{prediction:.1f} µg/m³")
            st.markdown(f"### {emoji} &nbsp; **{cat}**")
            st.markdown("*(China GB3095-2012 Standard)*")
            st.markdown("---")
            st.markdown(f"""| Reference | Limit |\n|-----------|-------|\n| China Grade II (annual) | 35 µg/m³ |\n| China Grade II (24-hour) | 75 µg/m³ |\n| WHO Annual Guideline | 5 µg/m³ |""")
        with col_gauge:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",value=prediction,
                title={"text":"Predicted PM2.5 (µg/m³)","font":{"size":15}},
                number={"suffix":" µg/m³","font":{"size":22}},
                gauge={"axis":{"range":[0,250],"tickwidth":1},
                       "steps":[{"range":[0,35],"color":"#27AE60"},{"range":[35,75],"color":"#F39C12"},
                                 {"range":[75,115],"color":"#E67E22"},{"range":[115,150],"color":"#E74C3C"},
                                 {"range":[150,250],"color":"#8E44AD"}],
                       "bar":{"color":"#2C3E50","thickness":0.25},
                       "threshold":{"line":{"color":"red","width":3},"thickness":0.75,"value":75}}
            ))
            fig_g.update_layout(height=300,margin=dict(t=40,b=10,l=20,r=20))
            st.plotly_chart(fig_g,use_container_width=True)
        st.markdown(f'<div class="inference-box">💡 <b>Result:</b> Predicted <b>{prediction:.1f} µg/m³</b> PM2.5 — classified as <b>{cat}</b> (China GB3095-2012). {"Exceeds" if prediction>75 else "Within"} the China 24-hour Grade II limit of 75 µg/m³.</div>',unsafe_allow_html=True)

# ── MODEL REPORT ──────────────────────────────────────────────────────────────
elif page == "📋  Model Report":
    st.title("📋 Model Performance Report")
    st.markdown("---")
    perf_path = "models/model_performance.csv"
    if os.path.exists(perf_path):
        st.subheader("Model Comparison")
        st.dataframe(pd.read_csv(perf_path), hide_index=True, use_container_width=True)
    else:
        st.info("Run the notebook to generate model_performance.csv")
    st.markdown("---")
    st.subheader("Configuration")
    st.markdown("""
| Parameter | Value |
|-----------|-------|
| **Algorithm** | Random Forest Regressor |
| **Target** | PM2.5 (µg/m³) |
| **Train/Test** | 80% / 20% — chronological |
| **Tuning** | GridSearchCV, 3-fold CV |
| **Metrics** | MAE, RMSE, R² |

**Features:** PM10, SO2, NO2, CO, O3, Temperature, Pressure, Dew Point, Rainfall, Wind Speed, RH, Hour, Month, Day of Week, Weekend flag, Zone Type
""")
    st.markdown("---")
    st.subheader("Saved Figures")
    fig_dir = "outputs"
    if os.path.exists(fig_dir):
        figs = sorted([f for f in os.listdir(fig_dir) if f.endswith(".png")])
        if figs:
            cols = st.columns(2)
            for i,ff in enumerate(figs):
                cols[i%2].image(os.path.join(fig_dir,ff),
                                caption=ff.replace("_"," ").replace(".png",""),
                                use_container_width=True)
        else:
            st.info("No PNG figures found. Run the notebook first.")
    else:
        st.info("outputs/ folder not found. Run the notebook first.")
