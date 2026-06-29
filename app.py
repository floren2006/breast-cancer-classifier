import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Klasifikasi Kanker Payudara", layout="wide")
st.title("🩺 Klasifikasi Kanker Payudara")
st.markdown("**Model Logistic Regression** — Sistem Pendukung Diagnosis Awal")

# Load model dan benchmark
@st.cache_resource
def load_model():
    return joblib.load("best_model_logreg.pkl")

@st.cache_data
def load_benchmark():
    return joblib.load("benchmark_stats.pkl")

model = load_model()
bench = load_benchmark()

mean_benign = bench['mean_benign']
mean_malignant = bench['mean_malignant']
feature_names = bench['feature_names']

st.sidebar.header("Input Fitur Pasien")
st.sidebar.markdown("Masukkan nilai 30 fitur sel (hasil FNA)")

# Input fitur (gunakan slider)
input_data = {}
default_vals = (mean_benign + mean_malignant) / 2

for feat in feature_names:
    min_val = float(min(mean_benign[feat], mean_malignant[feat]) * 0.5)
    max_val = float(max(mean_benign[feat], mean_malignant[feat]) * 1.5)
    default = float(default_vals[feat])
    input_data[feat] = st.sidebar.slider(
        f"{feat}",
        min_value=min_val,
        max_value=max_val,
        value=default,
        step=0.001,
        format="%.3f"
    )

# Tombol prediksi
if st.sidebar.button("🔍 Prediksi"):
    input_df = pd.DataFrame([input_data])
    pred_class = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Hasil Prediksi")
        if pred_class == 1:
            st.error("⚠️ **Malignant (Ganas)**")
        else:
            st.success("✅ **Benign (Jinak)**")
        st.write(f"Probabilitas Benign : {pred_prob[0]:.2%}")
        st.write(f"Probabilitas Malignant : {pred_prob[1]:.2%}")

    with col2:
        st.subheader("📈 Perbandingan dengan Rata-rata Historis")
        top_features = ['radius_mean', 'texture_mean', 'perimeter_mean',
                        'area_mean', 'concave points_mean']
        comp_df = pd.DataFrame({
            'Fitur': top_features,
            'Pasien': [input_data[f] for f in top_features],
            'Rata-rata Benign': [mean_benign[f] for f in top_features],
            'Rata-rata Malignant': [mean_malignant[f] for f in top_features]
        })
        st.dataframe(comp_df.style.format({
            'Pasien': '{:.3f}',
            'Rata-rata Benign': '{:.3f}',
            'Rata-rata Malignant': '{:.3f}'
        }))

    # Insight 1: Feature Importance
    st.markdown("---")
    st.subheader("🔍 Insight 1: Feature Importance (Koefisien Model)")
    coef = model.named_steps['clf'].coef_[0]
    coef_df = pd.DataFrame({
        'Fitur': feature_names,
        'Koefisien': coef
    }).sort_values('Koefisien', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = 10
    top_coef = coef_df.head(top_n)
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in top_coef['Koefisien']]
    ax.barh(top_coef['Fitur'], top_coef['Koefisien'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Koefisien')
    ax.set_title('Top 10 Fitur Paling Berpengaruh')
    st.pyplot(fig)

    # Insight 2: Decision Space
    st.subheader("📌 Insight 2: Decision Space (radius_mean vs texture_mean)")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(mean_benign['radius_mean'], mean_benign['texture_mean'],
                color='#2ecc71', s=100, label='Rata-rata Benign', alpha=0.7)
    ax2.scatter(mean_malignant['radius_mean'], mean_malignant['texture_mean'],
                color='#e74c3c', s=100, label='Rata-rata Malignant', alpha=0.7)
    ax2.scatter(input_data['radius_mean'], input_data['texture_mean'],
                color='gold', s=200, marker='*', label='Pasien Baru', edgecolors='black')
    ax2.set_xlabel('radius_mean')
    ax2.set_ylabel('texture_mean')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # Insight 3: Patient Benchmarking
    st.subheader("📊 Insight 3: Patient Benchmarking")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(top_features))
    width = 0.25
    vals_patient = [input_data[f] for f in top_features]
    vals_benign = [mean_benign[f] for f in top_features]
    vals_malignant = [mean_malignant[f] for f in top_features]

    ax3.bar(x - width, vals_benign, width, label='Rata-rata Benign', color='#2ecc71')
    ax3.bar(x, vals_malignant, width, label='Rata-rata Malignant', color='#e74c3c')
    ax3.bar(x + width, vals_patient, width, label='Pasien Baru', color='#f39c12')
    ax3.set_xticks(x)
    ax3.set_xticklabels(top_features, rotation=15, ha='right')
    ax3.set_ylabel('Nilai')
    ax3.legend()
    st.pyplot(fig3)

    # Insight 4: Texture vs Smoothness
    st.subheader("🧬 Insight 4: Texture vs Smoothness")
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.scatter(mean_benign['texture_mean'], mean_benign['smoothness_mean'],
                color='#2ecc71', s=100, label='Rata-rata Benign', alpha=0.7)
    ax4.scatter(mean_malignant['texture_mean'], mean_malignant['smoothness_mean'],
                color='#e74c3c', s=100, label='Rata-rata Malignant', alpha=0.7)
    ax4.scatter(input_data['texture_mean'], input_data['smoothness_mean'],
                color='gold', s=200, marker='*', label='Pasien Baru', edgecolors='black')
    ax4.set_xlabel('texture_mean')
    ax4.set_ylabel('smoothness_mean')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)

    # Insight 5: Konvergensi
    st.subheader("📈 Insight 5: Konvergensi Performa Model")
    metrics = {
        'Accuracy': 0.9737,
        'Precision': 0.9756,
        'Recall': 0.9524,
        'F1 Score': 0.9639,
        'ROC-AUC': 0.9964
    }
    metrics_df = pd.DataFrame({
        'Metrik': list(metrics.keys()),
        'Nilai': [f"{v:.2%}" for v in metrics.values()]
    })
    st.table(metrics_df)

    st.markdown("---")
    st.subheader("📋 Ringkasan Model")
    st.markdown(f"""
    - **Model Terbaik**: GridSearchCV (Logistic Regression)
    - **Recall** : {metrics['Recall']:.2%}
    - **F1 Score**: {metrics['F1 Score']:.2%}
    - **ROC-AUC** : {metrics['ROC-AUC']:.2%}
    - **Jumlah Fitur**: {len(feature_names)}
    """)

else:
    st.info("👈 Masukkan nilai fitur di sidebar, lalu klik 'Prediksi'.")

st.markdown("---")
st.caption("Dibuat untuk Final Project Machine Learning — Klasifikasi Kanker Payudara")
