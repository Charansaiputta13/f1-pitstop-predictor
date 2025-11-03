# 🏎️ Formula 1 Pit Stop Strategy Predictor  
**Machine Learning + FastF1 + Streamlit**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![FastF1](https://img.shields.io/badge/FastF1-Telemetry-green?logo=formula1)](https://docs.fastf1.dev/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📘 Project Overview

The **Formula 1 Pit Stop Strategy Predictor** is a data-driven ML web app that analyzes **real race telemetry** and predicts **optimal pit stop laps** for F1 drivers.

Built using:
- 🧠 **Machine Learning (Random Forest Classifier)**
- ⚙️ **FastF1 API for telemetry data**
- 🌐 **Streamlit for visualization**
- 📊 **Scikit-learn for predictive modeling**

This project demonstrates **sports analytics**, **data preprocessing**, and **model deployment** — all wrapped in a clean, interactive dashboard.

---

## 🧩 Tech Stack

| Category | Tools |
|-----------|--------|
| Data Source | [FastF1 API](https://docs.fastf1.dev/) |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn, Streamlit |
| Deployment | Streamlit Cloud |
| Serialization | Joblib |

---

## ⚙️ Features

✅ Fetches real F1 telemetry data (laps, stints, compounds)  
✅ Performs feature engineering for race dynamics  
✅ Predicts next-lap pit stops using ML models  
✅ Interactive Streamlit dashboard with plots  
✅ Ready for cloud deployment via Streamlit  

---

## 🧠 How It Works

1. **Data Loader (`data_loader.py`)**  
   Fetches and caches F1 race data using FastF1.

2. **EDA Notebook (`notebooks/exploratory_analysis.ipynb`)**  
   Analyzes tire wear, stint length, and lap time patterns.

3. **Feature Engineering (`feature_engineering.py`)**  
   Extracts lap deltas, compound encodings, and tire degradation rates.

4. **Model Training (`model_training.py`)**  
   Trains a RandomForest model to predict pit stops.

5. **Streamlit App (`app.py`)**  
   Interactive web dashboard for live race visualization and predictions.

---

## 🧾 Project Structure

