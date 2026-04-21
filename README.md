# 🌏 Beijing Air Quality Analysis — CMP7005 PRAC1

**Student ID:** st20341331
**Module:** CMP7005 — Programming for Data Analysis 
**Module Leader:** Dr. Amrita Prasad
**Assessment:** PRAC1 — From Data to Application Development (70%)

---

## Project Overview

This project works through a complete data science pipeline using hourly 
air quality data collected from four monitoring stations in Beijing, 
covering the period March 2013 to February 2017.

| Task | Description | Weight |
|------|-------------|--------|
| **Task 1** | Data Selection and Handling | 5% |
| **Task 2** | Exploratory Data Analysis with 25 visualisations | 50% |
| **Task 3** | Random Forest PM2.5 Prediction Model | 15% |
| **Task 4** | Streamlit Interactive Application | 20% |
| **Task 5** | GitHub Version Control | 10% |

---

## Live Application

The Streamlit dashboard is live and can be accessed at any time here:

**https://cmp7005prac1st20341331.streamlit.app**

---

## Stations Selected

Four stations were chosen to compare urban and suburban air quality 
across Beijing, following the classification used by Xu and Zhang (2004) 
and Yao et al. (2015).

| Station | Type | District | Reason for Selection |
|---------|------|----------|----------------------|
| **Nongzhanguan** | 🔴 Urban | Chaoyang | Busy road junctions, high traffic emissions |
| **Wanshouxigong** | 🔴 Urban | Fengtai | Mixed residential and industrial area |
| **Shunyi** | 🔵 Suburban | Shunyi | Lower traffic, more agricultural surroundings |
| **Dingling** | 🔵 Suburban | Changping | Semi-rural, very little industrial activity |

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
│   └── model_performance.csv
└── outputs/
└── fig01 to fig25 saved figures
```

---
Note: The trained model file rf_pm25_model.pkl is not stored here 
because it is over GitHub's 100MB file size limit. It gets created 
automatically when you run the notebook from top to bottom.

---

## How to Run

### Open in Google Colab
1. Upload the 4 CSV files into the data/ folder
2. Open the notebook in Google Colab
3. Add your GitHub credentials at the top
4. Click Runtime then Run All

### Live Streamlit App
Already running at:
https://cmp7005prac1st20341331.streamlit.app

### Run Locally
git clone https://github.com/Amjadkhan-CMU/CMP7005_PRAC1_BeijingAQ.git
cd CMP7005_PRAC1_BeijingAQ
pip install -r requirements.txt
streamlit run app.py
---

## Key Findings

1. Urban stations recorded **17% higher mean PM2.5** compared to 
suburban stations, with averages of 84.9 vs 72.7 µg/m³
2. Winter had the highest pollution with a mean PM2.5 of **95.9 µg/m³**, 
which is 52% higher than summer, mainly because of coal heating and 
cold air trapping pollution near the ground
3. Ozone behaved differently to all other pollutants and peaked in 
summer through photochemical reactions in sunlight
4. The Random Forest model achieved an **R² above 0.90** for PM2.5 
prediction, with PM10 and CO coming out as the strongest predictors

---

## References

- Brauer et al. (2021) *NEJM*, 384(4).
- Li et al. (2024) *Atmospheric Environment*, 310.
- Xu and Zhang (2004) *Atmospheric Environment*, 38(26).
- Xu and Zhang (2020) *Science of The Total Environment*, 739.
- Yao et al. (2015) *Atmospheric Research*, 167.
