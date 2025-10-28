import joblib
import pandas as pd
import streamlit as st

# Load model dan scaler
model = joblib.load('model_churn.pkl')
scaler = joblib.load('scaler.pkl')

# Fitur yang digunakan
features = [
    'achieve_status',
    'company_tenure_years',
    'distance_to_office_km',
    'working_hours_per_week',
    'manager_support_score',
    'marital_status',
    'target_achievement',
    'promotion_potential'
]

st.title("💼 Employee Churn Prediction App")
st.write("Silakan isi data berikut untuk memprediksi apakah karyawan berpotensi churn:")
st.divider()

# Input user
user_input = {}
for f in features:
    if f == 'marital_status':
           # Dropdown pilihan status pernikahan
        user_input[f] = st.selectbox(
            "Marital Status",
            options=["Single", "Married", "Divorced"],  # ubah sesuai data asli kamu
            index=0
        )
    else:
    user_input[f] = st.number_input(f"{f.replace('_', ' ').title()}", value=0.0)

marital_map = {"Single": 0, "Married": 1}
user_input['marital_status'] = marital_map[user_input['marital_status']]

# Convert ke DataFrame
input_df = pd.DataFrame([user_input])

# Scaling
input_scaled = scaler.transform(input_df)

# Prediksi
if st.button("Prediksi Churn"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    st.write(f"### Hasil: {'Churn' if pred==1 else 'Tidak Churn'}")
    st.write(f"Probabilitas churn: {prob:.2f}")
    if prob > 0.5:
        st.warning("Karyawan ini berisiko tinggi untuk churn.")
    else:
        st.success("Karyawan ini berisiko rendah untuk churn.")