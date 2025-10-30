import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Load model
churn_model = joblib.load("churn_model.pkl")
churn_period_model = joblib.load("churn_period_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💼 Employee Churn & Period Prediction")

# Input fields
target_achievement = st.number_input("🎯 Target Achievement (rasio)", 0.0, 2.0, 1.0, 0.01)
company_tenure_years = st.number_input("🧭 Lama bekerja (tahun)", 0.0, 40.0, 3.0, 0.1)
distance_to_office_km = st.number_input("📍 Jarak ke kantor (km)", 0.0, 60.0, 5.0, 0.1)
job_satisfaction = st.slider("😊 Kepuasan kerja (1-5)", 1, 5, 3)
manager_support_score = st.slider("🤝 Dukungan manajer (1-5)", 1, 5, 3)
marital_status = st.selectbox("💍 Status Pernikahan", ["Married", "Single"])
working_hours_per_week = st.number_input("⌚ Jam kerja/minggu", 20, 80, 40)

# Buat dataframe input
input_df = pd.DataFrame([{
    "target_achievement": target_achievement,
    "company_tenure_years": company_tenure_years,
    "distance_to_office_km": distance_to_office_km,
    "job_satisfaction": job_satisfaction,
    "manager_support_score": manager_support_score,
    "marital_status": 0 if marital_status == "Married" else 1,
    "working_hours_per_week": working_hours_per_week
}])

# Fitur turunan
input_df["achieve_status"] = input_df["target_achievement"]
input_df["promotion_potential"] = (
    (input_df["job_satisfaction"] + 2 * input_df["target_achievement"]) * np.log(input_df["company_tenure_years"] + 1)
)

# Urutan sesuai model
selected_features = [
    "achieve_status", "company_tenure_years", "distance_to_office_km",
    "working_hours_per_week", "manager_support_score",
    "target_achievement", "promotion_potential", "marital_status"
]

X_input = input_df[selected_features].copy()

# Pastikan urutan dan nama kolom sesuai dengan scaler yang digunakan saat training
if hasattr(scaler, "feature_names_in_"):
    trained_features = list(scaler.feature_names_in_)
    X_input = X_input.reindex(columns=trained_features, fill_value=0)

# Scaling
X_scaled = scaler.transform(X_input)

# Prediksi
if st.button("🔮 Jalankan Prediksi"):
    churn_pred = churn_model.predict(X_scaled)[0]
    if churn_pred == 0:
        st.success("📊 Karyawan TIDAK AKAN CHURN")
    else:
        st.warning("📊 Karyawan AKAN CHURN")
        churn_period_pred = churn_period_model.predict(X_scaled)[0]
        churn_label = {1: "Onboarding", 2: "1 Month", 3: "3 Months"}.get(churn_period_pred, "Unknown")
        st.write(f"⏳ Periode Churn yang diprediksi: **{churn_label}**")
