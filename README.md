# 🌏 Beijing Air Quality Analysis — CMP7005 PRAC1

**Student ID:** st20341331 

**Module:** CMP7005 — Programming for Data Analysis 

**Module Leader:** Dr. Amrita Prasad

**Assessment:** PRAC1 — From Data to Application Development (70%)

---

## Project Overview

This project works through a complete data science pipeline using hourly 
air quality data collected from four monitoring stations in Beijing, 
covering the period March 2013 to February 2017. The work is broken 
down into five tasks:

- **Task 1** — Loading and merging the four station datasets into one
- **Task 2** — Exploring and analysing the data through charts and statistics
- **Task 3** — Building a Random Forest model to predict PM2.5 levels
- **Task 4** — Developing an interactive Streamlit web application
- **Task 5** — Managing the project using GitHub version control

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

Batterman, S. et al. (2016) 'Air pollution and health impacts in
Beijing', *Environmental Health Perspectives*, 124(6), pp. 961–970.

Brauer, M. et al. (2021) 'Ambient particulate matter air pollution
and cardiovascular disease', *New England Journal of Medicine*,
384(4), pp. 388–390.

Li, X. et al. (2019) 'Long-term exposure to PM2.5 and lung cancer
risk', *International Journal of Environmental Research and Public
Health*, 16(14), p. 2477.

Li, Y. et al. (2024) 'Recent improvements in Beijing air quality:
trends and contributing drivers', *Atmospheric Environment*,
310, p. 119965.

Lim, S. et al. (2020) 'A comparative risk assessment of the burden
of disease attributable to air pollution', *The Lancet*, 380(9849),
pp. 2224–2260.

Sokhi, R. et al. (2022) 'A global observational analysis to
understand changes in air quality during exceptionally low
anthropogenic emission conditions', *Environment International*,
157, p. 106818.

Xu, J. and Zhang, Y. (2004) 'Spatial and temporal patterns of
PM2.5 at multiple urban and suburban stations in Beijing',
*Atmospheric Environment*, 38(26), pp. 4295–4306.

Xu, J. and Zhang, Y. (2020) 'Air quality trends in Beijing over
2013–2018', *Science of The Total Environment*, 739, p. 140032.

Yao, L. et al. (2015) 'Characteristics of PM2.5 in Beijing: mass
concentrations, chemical compositions, seasonal variations and
sources', *Atmospheric Research*, 167, pp. 62–72.


**Challenges Faced During This Project**

So coming from a Mechanical Engineering background, this was only my second semester at university. I had used GitHub before and done some basic coding in my first semester but this was genuinely my first time doing proper data analysis. So I was not completely new to programming but working with a real world dataset with over 140,000 rows was a completely different experience to anything I had done before.

**Working with Large Real World Data**

In my first semester the coding was mostly small structured exercises where the data was already clean and ready to go. This project was the first time I actually had to deal with a proper messy real world dataset. Missing values, inconsistent formatting, multiple files that needed merging together. Figuring out how to handle all of that in a systematic way rather than just patching things one by one took a lot more thinking than I expected honestly.
Understanding the Data Structure
When I first opened the CSV files I saw that the date and time were split across four separate columns, year, month, day and hour. I had never built a datetime index from scratch before and it took me a while to understand why it even mattered for the time series analysis later and how to do it properly inside a reusable function.

**Missing Values**

This was something I had never really dealt with before. In engineering if a value is missing you just go back and measure it again. With a dataset this big that is just not possible. I had to figure out the difference between time based interpolation and forward fill and understand which one made more sense for pollutants versus meteorological variables. That took quite a bit of reading and trial and error before I actually felt confident with what I was doing.

**Choosing the Right Visualisation**

This was harder than I thought it would be. In engineering we mostly just use line graphs and bar charts. For this project I had to get my head around heatmaps, pairplots, sunburst charts and box plots and actually understand when each one adds value rather than just making the data look different. Getting the charts to tell a proper story took more effort than I expected.

**Machine Learning**

This was honestly the hardest part of the whole project for me. I had some basic statistics knowledge from my engineering background but machine learning concepts like train test splits, cross validation, hyperparameter tuning and feature importance were all new to me in a practical sense. I spent a lot of time understanding why Random Forest was a better fit than Linear Regression for this dataset before I felt comfortable enough to properly justify that decision in the notebook.

**GitHub**

I had used GitHub in my first semester but only for basic stuff like uploading files and making simple commits. This project was the first time I actually used it as part of a proper ongoing workflow, committing at different stages with meaningful messages. There were a few issues along the way, file size limits, authentication errors, conflicts between local and remote versions. Working through all of that gave me a much better understanding of version control than I had before.

**Deploying the Streamlit App**

Building the Streamlit app was something I actually really enjoyed once I got into it. But getting it deployed on Streamlit Cloud and making sure it could read the data and model files correctly was not straightforward at all. File path issues, missing dependencies, the trained model file being too large for GitHub. These are the kind of problems that never show up in tutorials but they come up all the time in real development work.

Overall this was the most technically demanding thing I have done so far at university. Coming from mechanical engineering and doing data analysis properly for the first time, the learning curve at the start was steep. But working through each problem one by one gave me a level of practical programming and data science experience that I think is genuinely going to be useful going forward.
