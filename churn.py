import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# =====================================
# 💾 MUAT MODEL DAN SCALER
# =====================================
@st.cache_resource
def load_models():
    churn_model = joblib.load("churn_model.pkl")
    churn_period_model = joblib.load("churn_period_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return churn_model, churn_period_model, scaler

churn_model, churn_period_model, scaler = load_models()

# =====================================
# 🧩 FUNGSI PREPROCESS DATA
# =====================================
def preprocess_employee_data(df):
    df = df.copy()
    df['achieve_status'] = df['target_achievement']
    df['promotion_potential'] = (
        (df['job_satisfaction'] + 2 * df['target_achievement'])
        * np.log(df['company_tenure_years'] + 1)
    )
    df['marital_status'] = df['marital_status'].apply(lambda x: 0 if str(x).lower() == 'married' else 1)

    selected_features = [
        "achieve_status", "company_tenure_years", "distance_to_office_km",
        "working_hours_per_week", "manager_support_score",
        "target_achievement", "promotion_potential", "marital_status"
    ]

    X = df[selected_features].copy()

    # pastikan urutan fitur sesuai scaler saat training
    if hasattr(scaler, "feature_names_in_"):
        X = X.reindex(columns=list(scaler.feature_names_in_), fill_value=0)

    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

# =====================================
# ⚙️ FUNGSI PREDIKSI BATCH
# =====================================
def predict_churn_batch(df_raw):
    X_scaled = preprocess_employee_data(df_raw)
    churn_pred = churn_model.predict(X_scaled)
    churn_period_pred = [
        churn_period_model.predict(X_scaled.iloc[[i]])[0] if c == 1 else 0
        for i, c in enumerate(churn_pred)
    ]

    df_result = df_raw.copy()
    df_result["churn_pred"] = churn_pred
    df_result["churn_period_pred"] = churn_period_pred

    mapping = {0: "No Churn", 1: "Onboarding", 2: "1 Month", 3: "3 Months"}
    df_result["churn_period_label"] = df_result["churn_period_pred"].map(mapping)
    return df_result

# =====================================
# 🎨 STREAMLIT UI
# =====================================
st.set_page_config(page_title="Employee Churn Prediction", page_icon="💼", layout="centered")

st.title("💼 Employee Churn & Period Prediction Dashboard")

# Buat dua tab
tab1, tab2 = st.tabs(["🧍 Prediksi Individu", "📂 Prediksi Batch"])

# =====================================
# 🧍 TAB 1: Prediksi Individu
# =====================================
with tab1:
    st.subheader("Prediksi Churn untuk 1 Karyawan")

    col1, col2 = st.columns(2)

    with col1:
        target_achievement = st.number_input("🎯 Target Achievement (rasio)", 0.0, 2.0, 1.0, 0.01)
        company_tenure_years = st.number_input("🧭 Lama bekerja (tahun)", 0.0, 40.0, 3.0, 0.1)

    with col2:
        distance_to_office_km = st.number_input("📍 Jarak ke kantor (km)", 0.0, 60.0, 5.0, 0.1)
        working_hours_per_week = st.number_input("⌚ Jam kerja/minggu", 20, 80, 40)

    marital_status = st.selectbox("💍 Status Pernikahan", ["Married", "Single"])

    job_satisfaction = st.slider("😊 Kepuasan kerja (1-5)", 1, 5, 3)
    manager_support_score = st.slider("🤝 Dukungan manajer (1-5)", 1, 5, 3)
    
    input_df = pd.DataFrame([{
        "target_achievement": target_achievement,
        "company_tenure_years": company_tenure_years,
        "distance_to_office_km": distance_to_office_km,
        "job_satisfaction": job_satisfaction,
        "manager_support_score": manager_support_score,
        "marital_status": marital_status,
        "working_hours_per_week": working_hours_per_week
    }])

    if st.button("🔮 Jalankan Prediksi Individu"):
        X_scaled = preprocess_employee_data(input_df)
        churn_pred = churn_model.predict(X_scaled)[0]

        if churn_pred == 0:
            st.success("📊 Prediksi: Karyawan **TIDAK AKAN CHURN**")
        else:
            st.warning("📊 Prediksi: Karyawan **AKAN CHURN**")
            churn_period_pred = churn_period_model.predict(X_scaled)[0]
            churn_label = {1: "Onboarding", 2: "1 Month", 3: "3 Months"}.get(churn_period_pred, "Unknown")
            st.write(f"⏳ Periode Churn yang Diprediksi: **{churn_label}**")

# =====================================
# 📂 TAB 2: Prediksi Batch
# =====================================
with tab2:
    st.subheader("Prediksi Banyak Karyawan (Batch Upload)")
    uploaded_file = st.file_uploader("📁 Upload file CSV / Excel", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ File '{uploaded_file.name}' berhasil dimuat! Jumlah baris: {len(df)}")

            if st.button("🚀 Jalankan Prediksi Batch"):
                df_result = predict_churn_batch(df)
                st.success("✅ Prediksi batch berhasil dijalankan!")

                # 📊 Grafik
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                churn_counts = df_result['churn_pred'].value_counts().rename({0: 'No Churn', 1: 'Churn'})
                axes[0].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
                axes[0].set_title("Distribusi Prediksi Churn")

                period_counts = df_result['churn_period_label'].value_counts()
                axes[1].bar(period_counts.index, period_counts.values)
                axes[1].set_title("Distribusi Periode Churn")
                axes[1].set_ylabel("Jumlah Karyawan")
                axes[1].set_xticklabels(period_counts.index, rotation=15)
                st.pyplot(fig)

                st.dataframe(df_result.head(10))

        except Exception as e:
            st.error(f"🚨 Terjadi kesalahan saat membaca atau memproses file: {e}")
