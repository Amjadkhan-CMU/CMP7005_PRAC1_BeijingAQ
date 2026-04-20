# 🌏 Beijing Air Quality Analysis — CMP7005 PRAC1

**Cardiff Metropolitan University | School of Technologies**  
**Module:** CMP7005 — Programming for Data Analysis | **Semester 2, 2025–26**  
**Assessment:** PRAC1 — From Data to Application Development (70%)

---

## Project Overview

A complete data science pipeline applied to hourly air quality data from **four Beijing monitoring stations**, March 2013 – February 2017.

| Task | Description | Weight |
|------|-------------|--------|
| **Task 1** | Data Selection & Handling | 5% |
| **Task 2** | Exploratory Data Analysis (25 figures) | 50% |
| **Task 3** | Random Forest PM2.5 Prediction | 15% |
| **Task 4** | Streamlit Interactive Application | 20% |
| **Task 5** | GitHub Version Control (this repo) | 10% |

---

## Station Selection

| Station | Type | District | Justification |
|---------|------|----------|---------------|
| **Nongzhanguan** | 🔴 Urban | Chaoyang | Major arterial intersections, high traffic NOx |
| **Wanshouxigong** | 🔴 Urban | Fengtai | Mixed residential–industrial corridor |
| **Shunyi** | 🔵 Suburban | Shunyi | Lower traffic density, agricultural background |
| **Dingling** | 🔵 Suburban | Changping | Semi-rural, minimal industrial pressure |

*Classification follows Xu & Zhang (2004) and Yao et al. (2015).*

---

## Repository Structure

```
CMP7005_PRAC1_BeijingAQ/
├── CMP7005_PRAC1_Beijing_AirQuality.ipynb
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── PRSA_Nongzhanguan_2013_2017_Urban.csv
│   ├── PRSA_Wanshouxigong_2013_2017_Urban.csv
│   ├── PRSA_Shunyi_2013_2017_SubUrban.csv
│   ├── PRSA_Dingling_2013_2017_SubUrban.csv
│   ├── beijing_aq_raw_combined.csv
│   └── beijing_aq_clean.csv
├── models/
│   ├── rf_pm25_model.pkl
│   ├── scaler.pkl
│   └── model_performance.csv
└── outputs/
    └── fig01_*.png … fig25_*.png
```

---

## How to Run

### Notebook (Google Colab)
1. Upload the 4 CSV files to the repo's `data/` folder
2. Open `CMP7005_PRAC1_Beijing_AirQuality.ipynb` in Google Colab
3. Fill in GitHub credentials at the top
4. Run All — commits happen automatically at each task boundary

### Streamlit App (Local)
```bash
git clone https://github.com/YOUR_USERNAME/CMP7005_PRAC1_BeijingAQ.git
cd CMP7005_PRAC1_BeijingAQ
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit App (Google Colab)
```python
!pip install streamlit pyngrok -q
!streamlit run app.py &
from pyngrok import ngrok
print(ngrok.connect(8501))
```

---

## Key Findings

1. Urban stations record **17% higher mean PM2.5** than suburban stations (84.9 vs 72.7 µg/m³)
2. **Winter PM2.5 (95.9 µg/m³)** is 52% higher than summer (63.0 µg/m³) — coal heating + inversions
3. O3 **inversely** peaks in summer through photochemical formation
4. Southerly winds bring the most polluted air — transport from Hebei/Shandong provinces
5. Random Forest achieves **R² > 0.90** for PM2.5 prediction; PM10 and CO are top predictors

---

## Commit History

| # | Message | Task |
|---|---------|------|
| 1 | `Task 1: Load and merge 4 Beijing AQ station CSVs into unified dataset` | Task 1 |
| 2 | `Task 2a/2b: EDA data understanding + preprocessing — imputation, feature engineering` | Task 2 |
| 3 | `Task 2c: EDA analysis complete — univariate, bivariate, multivariate, 22 figures` | Task 2 |
| 4 | `Task 3: Random Forest PM2.5 predictor — GridSearchCV optimisation, feature importance` | Task 3 |
| 5 | `Task 4: Streamlit app — 5-section GUI` | Task 4 |
| 6 | `Final: Complete PRAC1 notebook — all 5 tasks` | All |

---

## References
- Brauer et al. (2021) *NEJM*, 384(4).
- Li et al. (2024) *Atmospheric Environment*, 310.
- Xu & Zhang (2004) *Atmospheric Environment*, 38(26).
- Xu & Zhang (2020) *Science of The Total Environment*, 739.
- Yao et al. (2015) *Atmospheric Research*, 167.
