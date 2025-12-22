 # Streamlit Code
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and label encoder 
model = joblib.load("aqi_model.pkl")          # trained on 9 features
le = joblib.load("label_encoder.pkl")         # for state column

# Load the final dataset
df = pd.read_csv("final_aqi_dataset.csv")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")

st.title("AQI PREDICTION AND YEAR-WISE COMPARISON APP")

# ---------------------- AQI PREDICTION ----------------------
st.header("🔷 Predict AQI using Pollutant Indices")

SOi = st.number_input("SOi Index", min_value=0.0, format="%.2f")
Noi = st.number_input("Noi Index", min_value=0.0, format="%.2f")
Rpi = st.number_input("Rpi Index", min_value=0.0, format="%.2f")
SPMi = st.number_input("SPMi Index", min_value=0.0, format="%.2f")
PM2_5i = st.number_input("PM2.5 Index", min_value=0.0, format="%.2f")

States = st.selectbox("Select State", list(le.classes_))

month = st.number_input("Month", min_value=1, max_value=12, step=1)
day   = st.number_input("Day", min_value=1, max_value=31, step=1)
year  = st.number_input("Year", min_value=1990, max_value=2015, step=1)

if st.button("Predict AQI"):
    try:
        state_encoded = le.transform([States])[0]

        # ---- CORRECT: Now using 9 features ----
        features = np.array([[SOi, Noi, Rpi, SPMi, PM2_5i,
                              state_encoded, int(month), int(day), int(year)]])

        prediction = model.predict(features)[0]
        st.success(f"Predicted AQI: {round(prediction, 2)}")

    except Exception as e:
        st.error(f"⚠ Error: {e}")

# ---------------------- YEAR COMPARISON ----------------------
st.header("🔶 Compare AQI of Two Different Years")

year_list = sorted(df["year"].dropna().astype(int).unique())
year1 = st.selectbox("Select Year 1", year_list)
year2 = st.selectbox("Select Year 2", year_list)

if st.button("Compare AQI of Selected Years"):
    data_y1 = df[df["year"] == year1]
    data_y2 = df[df["year"] == year2]

    if data_y1.empty or data_y2.empty:
        st.error("⚠ No data found for one of the selected years.")
    else:
        avg1 = data_y1["AQI"].mean()
        avg2 = data_y2["AQI"].mean()

        st.subheader("📊 Year-wise Average AQI")
        st.write(f"Average AQI for {year1}: **{round(avg1, 2)}**")
        st.write(f"Average AQI for {year2}: **{round(avg2, 2)}**")

        # ---- FIXED: removed invalid hue ----
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        sns.barplot(x=[year1], y=[avg1], ax=ax[0], palette="Set2")
        ax[0].set_title(f"AQI {year1}",fontsize=16,fontweight='bold')
        ax[0].set_ylabel("AQI",fontsize=14,fontweight='bold')

        sns.barplot(x=[year2], y=[avg2], ax=ax[1], palette="Accent")
        ax[1].set_title(f"AQI {year2}",fontsize=16,fontweight='bold')
        ax[1].set_ylabel("AQI",fontsize=14,fontweight='bold')

        st.pyplot(fig)
