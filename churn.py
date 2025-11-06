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
    churn_period_features = joblib.load("churn_period_features.pkl")
    scaler = joblib.load("scaler.pkl")
    return churn_model, churn_period_model, churn_period_features, scaler

churn_model, churn_period_model, churn_period_features, scaler = load_models()

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
# 🎨 STREAMLIT UI
# =====================================
st.set_page_config(page_title="Employee Churn Prediction", page_icon="💼", layout="wide")

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
            persona = "Committed Performer"
            recommendation = "Berikan peluang pengembangan karier agar tetap engaged dan loyal."
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
                 top_factors = shap_df.head(3)
                 top_risk_factor = ", ".join(top_factors["Feature"].tolist())
            except Exception as e:
                 abs_vals = np.abs(X_scaled.iloc[0].values)
                 top_idx = np.argsort(abs_vals)[-3:][::-1]
                 top_risk_factor = ", ".join(X_scaled.columns[top_idx])

            st.error(f"📊 Prediksi: Karyawan **AKAN CHURN** ({churn_status})")
        
        st.write(f"📈 Probabilitas churn: **{churn_prob*100:.1f}%**")
        st.info(f"⚠️ Tiga Faktor Utama karyawan resign: **{top_risk_factor}**")
        st.write(f"👤 Persona: {persona}")
        st.write(f"💡 Rekomendasi HR: {recommendation}")

# =====================================
# 📂 TAB 2: Prediksi Batch
# =====================================

# =====================================================
# 🔧 HELPER FUNCTIONS
# =====================================================
def _ensure_numeric_df(df):
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def align_for_model(X, model):
    if not hasattr(model, "feature_names_in_"):
        return X
    trained_feats = list(model.feature_names_in_)
    X_aligned = pd.DataFrame(0.0, index=X.index, columns=trained_feats)
    for col in trained_feats:
        if col in X.columns:
            X_aligned[col] = X[col].astype(float)
    return X_aligned

# =====================================================
# 🧩 PREPROCESSING
# =====================================================
def preprocess_batch_robust(df):
    df = df.copy()

    # Simpan kolom identitas (jika ada)
    id_cols = [c for c in ['employee_id', 'employee_name'] if c in df.columns]
    id_data = df[id_cols].copy() if id_cols else pd.DataFrame()

    # Cek kolom wajib
    required_cols = [
        'target_achievement', 'company_tenure_years', 'distance_to_office_km',
        'job_satisfaction', 'manager_support_score', 'marital_status', 'working_hours_per_week'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"⚠️ Kolom hilang dalam file input: {missing}")

    # Feature engineering
    df['achieve_status'] = df['target_achievement']
    df['promotion_potential'] = (
        (df['job_satisfaction'] + 2 * df['target_achievement']) * np.log(df['company_tenure_years'] + 1)
    )

    def map_marital(x):
        if pd.isna(x):
            return 0
        s = str(x).strip().lower()
        return 0 if s in ('married', 'm') else 1
    df['marital_status'] = df['marital_status'].apply(map_marital)

    selected_features = [
        'achieve_status', 'company_tenure_years', 'distance_to_office_km',
        'working_hours_per_week', 'manager_support_score', 'marital_status',
        'target_achievement', 'promotion_potential'
    ]

    X = _ensure_numeric_df(df[selected_features])

    # Align to scaler
    if hasattr(scaler, 'feature_names_in_'):
        trained_feats = list(scaler.feature_names_in_)
        X_aligned = pd.DataFrame(0.0, index=X.index, columns=trained_feats)
        for col in trained_feats:
            if col in X.columns:
                X_aligned[col] = X[col].astype(float)
        X = X_aligned

    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    if not id_data.empty:
        df_scaled = pd.concat([id_data.reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)

    return df_scaled

# =====================================================
# 🧩 PREPROCESS FOR churn_period_model
# =====================================================
def preprocess_for_churn_period(df_scaled):
    df_cp = df_scaled.copy()
    cp_feats = list(churn_period_features)
    for col in cp_feats:
        if col not in df_cp.columns:
            df_cp[col] = 0.0
    return df_cp.reindex(columns=cp_feats, fill_value=0.0).astype('float64')

# =====================================================
# ⚙️ PREDIKSI BERTINGKAT (SINKRON)
# =====================================================
def predict_batch_safe(df_raw):
    X_scaled = preprocess_batch_robust(df_raw)
    id_cols = [c for c in ['employee_id', 'employee_name'] if c in X_scaled.columns]
    
    # Hapus kolom ID/Nama sebelum prediksi
    X_for_churn = align_for_model(X_scaled.drop(columns=id_cols, errors='ignore'), churn_model)

    churn_pred = churn_model.predict(X_for_churn).astype(int)
    churn_period_pred = []

    for i, c in enumerate(churn_pred):
        if int(c) == 0:
            churn_period_pred.append(0)  # Stayed
            continue

        Xi_cp = preprocess_for_churn_period(X_scaled.iloc[[i]].copy())
        Xi_cp = align_for_model(Xi_cp, churn_period_model)

        try:
            p = int(churn_period_model.predict(Xi_cp)[0]) + 1  # shift 0→1, 1→2, 2→3
        except Exception:
            p = 1
        churn_period_pred.append(p)

    df_result = df_raw.copy().reset_index(drop=True)
    df_result['churn_pred'] = churn_pred
    df_result['churn_period_pred'] = churn_period_pred

    # Map period (0 = Stayed, 1=Onboarding, 2=1Month, 3=3Months)
    mapping = {0: "Stayed", 1: "Onboarding", 2: "1 Month", 3: "3 Months"}
    df_result['churn_period_label'] = df_result['churn_period_pred'].map(mapping)

    # Pastikan label konsisten
    df_result.loc[df_result['churn_pred'] == 0, 'churn_period_pred'] = 0
    df_result.loc[df_result['churn_pred'] == 0, 'churn_period_label'] = "Stayed"
    df_result.loc[(df_result['churn_pred'] == 1) & (df_result['churn_period_pred'] == 0), 'churn_period_pred'] = 1
    df_result.loc[(df_result['churn_pred'] == 1) & (df_result['churn_period_label'] == "Stayed"), 'churn_period_label'] = "Onboarding"

    df_result['churn_label_final'] = df_result['churn_pred'].map({0: "No Churn", 1: "Churn"})
    
    # Susun ulang kolom agar ID & nama tampil di depan
    if id_cols:
        cols = id_cols + [c for c in df_result.columns if c not in id_cols]
        df_result = df_result[cols]
    
    return df_result
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

            col1, col2 = st.columns([1, 1])
            FIGSIZE = (5, 4)

            with col1:
                fig1, ax1 = plt.subplots(figsize=FIGSIZE)
                ax1.pie(
                    churn_counts.values,
                    labels=churn_counts.index,
                    autopct="%1.1f%%",
                    colors=["#EF5350", "#66BB6A"],
                    startangle=90,
                    textprops={'fontsize': 10}
                )
                ax1.set_title("Distribusi Churn", fontsize=12, pad=10)
                ax1.axis('equal')
                fig1.tight_layout()
                st.pyplot(fig1, use_container_width=True)

            with col2:
                fig2, ax2 = plt.subplots(figsize=FIGSIZE)
                colors = ["#FFA726", "#FB8C00", "#F57C00", "#42A5F5"]
                ax2.bar(period_counts.index, period_counts.values, color=colors, width=0.6)
                ax2.set_title("Distribusi Periode Churn", fontsize=12, pad=10)
                ax2.set_ylabel("Jumlah Karyawan", fontsize=10)
                ax2.tick_params(axis='x', rotation=20)
                ax2.set_ylim(0, max(period_counts.values)*1.2)
                fig2.tight_layout()
                st.pyplot(fig2, use_container_width=True)

            # ================================
            # 🧾 RINGKASAN HASIL PREDIKSI
            # ================================
            st.subheader("📊 Ringkasan Hasil Prediksi Batch")

            try:
                total_karyawan = len(df_result)
                total_churn = df_result["churn_pred"].sum()
                persen_churn = (total_churn / total_karyawan) * 100
                periode_counts = df_result["churn_period_label"].value_counts()

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("👥 Total Karyawan", total_karyawan)
                col2.metric("🚪 Prediksi Churn", total_churn, f"{persen_churn:.1f}%")
                col3.metric("🧭 Rata-rata Tenure", f"{df_result['company_tenure_years'].mean():.1f} tahun")
                col4.metric("🔥 Periode Churn Tertinggi", periode_counts.idxmax() if not periode_counts.empty else "-")

                # ================================
                # 🧠 INTERPRETASI OTOMATIS
                # ================================
                st.subheader("🧠 Insight Otomatis HR")

                if persen_churn > 50:
                    st.error(f"Tingkat churn **sangat tinggi ({persen_churn:.1f}%)**. "
                             "Perlu perhatian pada faktor kepuasan kerja dan dukungan manajer.")
                elif persen_churn > 30:
                    st.warning(f"Tingkat churn **menengah ({persen_churn:.1f}%)**. "
                               f"Periode churn terbanyak: **{periode_counts.idxmax()}**. "
                               "Fokus pada retensi karyawan baru dan keseimbangan kerja.")
                else:
                    st.success(f"Tingkat churn **rendah ({persen_churn:.1f}%)**. "
                               "Strategi retensi tampaknya efektif — pertahankan faktor motivasi & komunikasi yang ada.")

                # ================================
                # 🎯 SEGMENTASI RISIKO
                # ================================
                st.subheader("🎯 Segmentasi Risiko Karyawan")
                
                try:
                    churn_probs = churn_model.predict_proba(preprocess_batch_robust(df_raw))[:, 1]
                    df_result["churn_prob"] = churn_probs

                    bins = [0, 0.33, 0.66, 1.0]
                    labels = ["Low", "Medium", "High"]
                    df_result["risk_level"] = pd.cut(df_result["churn_prob"], bins=bins, labels=labels)

                    risk_counts = df_result["risk_level"].value_counts().reindex(labels, fill_value=0)
                    fig_risk, ax_risk = plt.subplots(figsize=(5, 3.5))
                    ax_risk.bar(risk_counts.index, risk_counts.values, color=["#66BB6A", "#FFA726", "#EF5350"], width=0.6)
                    ax_risk.set_title("Distribusi Level Risiko Karyawan", fontsize=12, pad=10)
                    ax_risk.set_ylabel("Jumlah Karyawan", fontsize=10)
                    st.pyplot(fig_risk, use_container_width=True)

                    # Insight tambahan
                    if risk_counts["High"] > 0:
                        st.warning(f"⚠️ {risk_counts['High']} karyawan dikategorikan **risiko tinggi** untuk resign dalam waktu dekat.")
                except Exception:
                    st.info("Model tidak mendukung probabilitas churn (predict_proba).")

                # ================================
                # 💡 ANALISIS FAKTOR PENTING (RINGKAS)
                # ================================
                st.subheader("💡 Analisis Faktor Penting terhadap Churn")

                try:
                    faktor_mean = df_result.groupby("churn_label_final")[["job_satisfaction", "manager_support_score", "target_achievement"]].mean().T
                    fig_faktor, ax_faktor = plt.subplots(figsize=(5.5, 3.5))
                    faktor_mean.plot(kind="bar", ax=ax_faktor)
                    ax_faktor.set_title("Perbandingan Faktor Rata-rata: Churn vs No Churn", fontsize=12, pad=10)
                    ax_faktor.set_ylabel("Rata-rata Skor")
                    ax_faktor.tick_params(axis="x", rotation=0)
                    st.pyplot(fig_faktor, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Tidak dapat menampilkan analisis faktor: {e}")

            except Exception as e:
                st.warning(f"⚠️ Tidak dapat membuat ringkasan: {e}")

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
            # 🔎 PENCARIAN BERDASARKAN ID KARYAWAN
            # ================================
            st.subheader("🔎 Cari Hasil Prediksi Berdasarkan Employee ID")

            if 'employee_id' in df_result.columns:
                search_val = st.text_input("Masukkan Employee ID:")
                if search_val:
                    filtered = df_result[df_result['employee_id'].astype(str).str.contains(search_val, case=False, na=False)]
                    if len(filtered) > 0:
                        st.dataframe(filtered)
                    else:
                        st.warning("❌ Tidak ditemukan hasil untuk Employee ID tersebut.")
            else:
                st.info("Kolom 'employee_id' tidak ditemukan di file input.")

            # =====================================================
            # 💾 DOWNLOAD FILE HASIL PREDIKSI DALAM FORMAT EXCEL
            # =====================================================
            from io import BytesIO

            def to_excel(df):
                """Konversi DataFrame ke file Excel (XLSX) dalam memory."""
                output = BytesIO()
                try:
                    import openpyxl
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        df.to_excel(writer, index=False, sheet_name="Prediksi")
                except ImportError:
                    try: 
                        import xlsxwriter
                        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                            df.to_excel(writer, index=False, sheet_name="Prediksi")
                    except Exception:
                        df.to_csv(output, index=False)
                processed_data = output.getvalue()
                return processed_data

            # Tombol download Excel muncul di bawah tabel
            excel_data = to_excel(df_result)
            st.download_button(
                label="📥 Download Hasil Prediksi (Excel)",
                data=excel_data,
                file_name="hasil_prediksi_churn.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # ================================
            # 🔍 SHAP – ANALISIS FAKTOR RISIKO
            # ================================
            st.subheader("🔍 Analisis Faktor Risiko (SHAP)")

            try:
                # Gunakan data scaled untuk SHAP
                X_scaled = preprocess_batch_robust(df_raw)
                X_for_churn = align_for_model(X_scaled, churn_model)

                try:
                    final_model = churn_model.named_steps['model']
                except Exception:
                    final_model = churn_model[-1] if hasattr(churn_model, '__getitem__') else churn_model
                    
                # Pilih explainer sesuai tipe model
                model_type = type(final_model).__name__.lower()
                if "tree" in model_type or "forest" in model_type or "boost" in model_type:
                    explainer = shap.TreeExplainer(final_model)
                else:
                    explainer = shap.LinearExplainer(final_model, X_for_churn)

                shap_values = explainer(X_for_churn)

                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                shap_importance = pd.DataFrame({
                    "Feature": X_for_churn.columns,
                    "Mean |SHAP Value|": mean_abs_shap
                }).sort_values("Mean |SHAP Value|", ascending=False)

                top5 = shap_importance.head(5)
                st.write("Top 5 faktor risiko utama yang berkontribusi terhadap churn:")
                st.dataframe(top5)

                fig3, ax3 = plt.subplots(figsize=(6, 3.5))   
                shap.plots.bar(shap_values, show=False)
                plt.title("Fitur Paling Berpengaruh Terhadap Churn", fontsize=12, pad= 10)
                plt.tight_layout()
                st.pyplot(fig3, use_container_width=False)

            except Exception as e:
                st.warning(f"⚠️ Analisis SHAP tidak dapat dijalankan: {e}")

        except Exception as e:
            st.error(f"🚨 Gagal memproses file: {e}")

