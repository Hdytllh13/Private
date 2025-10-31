import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE
import shap

# =====================================
# 💾 MUAT MODEL DAN SCALER
# =====================================
class StrongHybridSampler:
        def __init__(self, random_state=42, target_factor=1.0, max_samples_per_class=None):
            self.random_state = random_state
            self.target_factor = target_factor
            self.max_samples_per_class = max_samples_per_class
        
        def _make_sampling_strategy(self, y):
            ctr = Counter(y)
            max_count = max(ctr.values())
            target = {}
            for cls, c in ctr.items():
                desired = int(max_count * self.target_factor)
                if desired <= c:
                    continue
                if self.max_samples_per_class:
                    desired = min(desired, self.max_samples_per_class)
                target[cls] = desired
            return target
        
        def fit_resample(self, X, y):
            sampling_strategy = self._make_sampling_strategy(y)
            if not sampling_strategy:
                return X, y
            try:
                sampler = BorderlineSMOTE(random_state=self.random_state, sampling_strategy=sampling_strategy)
                X_res, y_res = sampler.fit_resample(X, y)
            except Exception:
                sampler = SMOTEENN(random_state=self.random_state, sampling_strategy=sampling_strategy)
                X_res, y_res = sampler.fit_resample(X, y)
            return X_res, y_res


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
# 🔍 HR PERSONA & RECOMMENDATION MAPPING
# =====================================
persona_mapping = {
    "Stayed": {
        "persona": "Committed Performer",
        "recommendation": "Berikan peluang pengembangan karier agar tetap engaged dan loyal."
    },
    "Onboarding": {
        "persona": "Balanced Contributor",
        "recommendation": "Fokus pada onboarding dan mentoring selama 1–2 bulan pertama untuk memperkuat ikatan."
    },
    "1 Month": {
        "persona": "Cautious Explorer",
        "recommendation": "Perkuat komunikasi dan dukungan manajerial; pastikan workload dan ekspektasi realistis."
    },
    "3 Months": {
        "persona": "Disengaged Achiever",
        "recommendation": "Lakukan 1-on-1 untuk memahami motivasi; tawarkan rotasi peran atau tantangan baru."
    }
}

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
        churn_prob = churn_model.predict_proba(X_scaled)[0][1]

        if churn_pred == 0:
            churn_status = "Stayed"
            st.success("📊 Prediksi: Karyawan **TIDAK AKAN CHURN**")
            top_risk_factor = None
        else:
            try:
                period_pred = int(churn_period_model.predict(X_scaled)[0])
                period_labels = {0: "Onboarding", 1: "1 Month", 2: "3 Months"}
                churn_status = period_labels.get(period_pred, "Churn")
                persona = persona_mapping.get(churn_status, {}).get("persona", "Unknown")
                recommendation = persona_mapping.get(churn_status, {}).get("recommendation", "Tidak ada rekomendasi.")
            except Exception:
                churn_status = "Churn"
            
            # =====================================
            # 🔍 TOP RISK FACTOR MENGGUNAKAN SHAP
            # =====================================
            try:
                 explainer = shap.Explainer(churn_model, feature_names=X_scaled.columns)
                 shap_values = explainer(X_scaled)
                 shap_df = pd.DataFrame({
                      "Feature": X_scaled.columns,
                      "SHAP Value": shap_values.values[0],
                      "Value": X_scaled.iloc[0].values
                      }).sort_values("SHAP Value", key=abs, ascending=False)
                 top_risk_factor = shap_df.iloc[0]["Feature"]
                 st.warning(f"⚠️ Top Risk Factor karyawan berisiko resign: **{top_risk_factor}**")
            except Exception as e:
                 top_risk_factor = X_scaled.columns[np.argmax(np.abs(X_scaled.iloc[0].values))]

            st.error(f"📊 Prediksi: Karyawan **AKAN CHURN** ({churn_status})")
        
        st.write(f"📈 Probabilitas churn: **{churn_prob*100:.1f}%**")
        st.info(f"⚠️ Top Risk Factor karyawan resign: **{top_risk_factor}**")
        st.write(f"👤 Persona: {persona}")
        st.write(f"💡 Rekomendasi HR: {recommendation}")

# =====================================
# 📂 TAB 2: Prediksi Batch
# =====================================
with tab2:
    st.subheader("📂 Prediksi Banyak Karyawan (Batch Upload)")

    uploaded_file = st.file_uploader("📁 Upload file CSV / Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File berhasil dimuat! Jumlah data: {len(df_raw)} baris")

            # ================================
            # 🔍 PREDIKSI BATCH
            # ================================
            with st.spinner("⏳ Sedang memproses prediksi batch..."):
                df_result = predict_batch_safe(df_raw)

            st.success("✅ Prediksi batch selesai dan sinkron!")

            # ================================
            # 📊 VISUALISASI DISTRIBUSI
            # ================================
            churn_counts = df_result["churn_label_final"].value_counts().reindex(["Churn", "No Churn"]).fillna(0)
            period_counts = df_result["churn_period_label"].value_counts().reindex(["Onboarding", "1 Month", "3 Months", "Stayed"]).fillna(0)

            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots()
                ax1.pie(
                    churn_counts.values,
                    labels=churn_counts.index,
                    autopct="%1.1f%%",
                    colors=["#EF5350", "#66BB6A"],
                    startangle=90
                )
                ax1.set_title("Distribusi Churn")
                st.pyplot(fig1)

            with col2:
                fig2, ax2 = plt.subplots()
                colors = ["#FFA726", "#FB8C00", "#F57C00", "#42A5F5"]
                ax2.bar(period_counts.index, period_counts.values, color=colors)
                ax2.set_title("Distribusi Periode Churn")
                ax2.set_ylabel("Jumlah Karyawan")
                st.pyplot(fig2)

            # ================================
            # 📈 SAMPLE PREVIEW
            # ================================
            st.subheader("📋 Contoh Data Hasil Prediksi")
            sample_option = st.selectbox("Tampilkan data:", ["Baris awal", "Baris akhir", "Acak"], index=0)

            if sample_option == "Baris awal":
                st.dataframe(df_result.head(10))
            elif sample_option == "Baris akhir":
                st.dataframe(df_result.tail(10))
            else:
                st.dataframe(df_result.sample(min(10, len(df_result))))

            # ================================
            # 🔍 SHAP – ANALISIS FAKTOR RISIKO
            # ================================
            st.subheader("🔍 Analisis Faktor Risiko (SHAP)")

            try:
                # Gunakan data scaled untuk SHAP
                X_scaled = preprocess_batch_robust(df_raw)
                X_for_churn = align_for_model(X_scaled, churn_model)

                explainer = shap.Explainer(churn_model, feature_names=X_for_churn.columns)
                shap_values = explainer(X_for_churn)

                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                shap_importance = pd.DataFrame({
                    "Feature": X_for_churn.columns,
                    "Mean |SHAP Value|": mean_abs_shap
                }).sort_values("Mean |SHAP Value|", ascending=False)

                top5 = shap_importance.head(5)
                st.write("Top 5 faktor risiko utama yang berkontribusi terhadap churn:")
                st.dataframe(top5)

                fig3, ax3 = plt.subplots()
                shap.plots.bar(shap_values, show=False)
                st.pyplot(fig3)

            except Exception as e:
                st.warning(f"⚠️ Analisis SHAP tidak dapat dijalankan: {e}")

        except Exception as e:
            st.error(f"🚨 Gagal memproses file: {e}")

