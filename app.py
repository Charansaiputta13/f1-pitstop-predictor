import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fastf1
import joblib
import os

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="🏎️ F1 Pit Stop Strategy Predictor",
    page_icon="🏁",
    layout="wide"
)
st.title("🏎️ Formula 1 Pit Stop Strategy Predictor")
st.caption("Analyze race telemetry and predict optimal pit stops using Machine Learning")

# ----------------------------
# Load ML Model
# ----------------------------
MODEL_PATH = "model/pitstop_model.pkl"
SCALER_PATH = "model/scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Trained model or scaler not found. Please run `model_training.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Race Configuration 🏁")
year = st.sidebar.selectbox("Select Year", [2024, 2023, 2022, 2021])
gp = st.sidebar.text_input("Enter Grand Prix (e.g., Monaco, Silverstone):", "Monaco")

# ----------------------------
# Load Race Data
# ----------------------------
@st.cache_data
def load_race_data(year, gp):
    fastf1.Cache.enable_cache("data/raw")
    session = fastf1.get_session(year, gp, "R")
    session.load()
    laps = session.laps
    return laps, session

if st.sidebar.button("Load Race Data"):
    with st.spinner(f"Loading {gp} GP {year} data..."):
        try:
            laps, session = load_race_data(year, gp)
            st.success(f"✅ Loaded {len(laps)} laps from {gp} {year}")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

    # ----------------------------
    # Data Processing
    # ----------------------------
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps = laps[laps["LapTimeSeconds"].notna()]

    # Sidebar driver select
    driver = st.sidebar.selectbox("Select Driver", sorted(laps["Driver"].unique()))

    driver_laps = laps.pick_driver(driver)
    st.subheader(f"📊 Lap Time Trend – {driver}")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=driver_laps, x="LapNumber", y="LapTimeSeconds", hue="Compound", linewidth=2, ax=ax)
    plt.title(f"{driver} Lap Times by Tire Compound")
    plt.xlabel("Lap Number")
    plt.ylabel("Lap Time (s)")
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)

    # ----------------------------
    # Prepare Features for Prediction
    # ----------------------------
    compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
    driver_laps["CompoundCode"] = driver_laps["Compound"].map(compound_map).fillna(5)
    driver_laps["LapDelta"] = driver_laps["LapTimeSeconds"].diff()
    driver_laps["AvgLapInStint"] = (
        driver_laps.groupby(["Stint"])["LapTimeSeconds"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    driver_laps["StintStartLapTime"] = driver_laps.groupby(["Stint"])["LapTimeSeconds"].transform("first")
    driver_laps["TireDegradation"] = driver_laps["LapTimeSeconds"] - driver_laps["StintStartLapTime"]

    features = driver_laps[["LapNumber", "Stint", "CompoundCode", "LapDelta", "AvgLapInStint", "TireDegradation"]].fillna(0)
    features_scaled = scaler.transform(features)

    # ----------------------------
    # Make Predictions
    # ----------------------------
    preds = model.predict(features_scaled)
    driver_laps["PitPrediction"] = preds

    pit_laps = driver_laps[driver_laps["PitPrediction"] == 1]["LapNumber"].tolist()

    st.subheader("🔮 Predicted Pit Stop Laps")
    if len(pit_laps) > 0:
        st.success(f"Model predicts pit stops at laps: {pit_laps}")
    else:
        st.info("No upcoming pit stops predicted in current stint.")

    # ----------------------------
    # Visualization of Pit Predictions
    # ----------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=driver_laps, x="LapNumber", y="LapTimeSeconds", color="gray", label="Lap Time", ax=ax2)
    for pit in pit_laps:
        plt.axvline(x=pit, color="red", linestyle="--", alpha=0.7, label="Predicted Pit" if pit == pit_laps[0] else "")
    plt.title(f"{driver} – Predicted Pit Stop Laps ({gp} {year})")
    plt.xlabel("Lap Number")
    plt.ylabel("Lap Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    st.pyplot(fig2)
