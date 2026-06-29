import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(page_title="Klasifikasi Kanker Payudara", layout="wide")
st.title("🩺 Klasifikasi Kanker Payudara")
st.markdown("**Model Logistic Regression** — Sistem Pendukung Diagnosis Awal")

# ==================== LOAD MODEL & BENCHMARK ====================
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
feature_names = bench['feature_names']  # list nama asli dari dataset

# ==================== MAPPING NAMA FITUR ====================
feature_labels = {
    'radius_mean': 'Radius (Rata-rata)',
    'texture_mean': 'Tekstur (Rata-rata)',
    'perimeter_mean': 'Perimeter (Rata-rata)',
    'area_mean': 'Luas (Rata-rata)',
    'smoothness_mean': 'Kehalusan (Rata-rata)',
    'compactness_mean': 'Kekompakan (Rata-rata)',
    'concavity_mean': 'Cekungan (Rata-rata)',
    'concave points_mean': 'Titik Cekung (Rata-rata)',
    'symmetry_mean': 'Simetri (Rata-rata)',
    'fractal_dimension_mean': 'Dimensi Fraktal (Rata-rata)',
    'radius_se': 'Radius (Std. Error)',
    'texture_se': 'Tekstur (Std. Error)',
    'perimeter_se': 'Perimeter (Std. Error)',
    'area_se': 'Luas (Std. Error)',
    'smoothness_se': 'Kehalusan (Std. Error)',
    'compactness_se': 'Kekompakan (Std. Error)',
    'concavity_se': 'Cekungan (Std. Error)',
    'concave points_se': 'Titik Cekung (Std. Error)',
    'symmetry_se': 'Simetri (Std. Error)',
    'fractal_dimension_se': 'Dimensi Fraktal (Std. Error)',
    'radius_worst': 'Radius (Terburuk)',
    'texture_worst': 'Tekstur (Terburuk)',
    'perimeter_worst': 'Perimeter (Terburuk)',
    'area_worst': 'Luas (Terburuk)',
    'smoothness_worst': 'Kehalusan (Terburuk)',
    'compactness_worst': 'Kekompakan (Terburuk)',
    'concavity_worst': 'Cekungan (Terburuk)',
    'concave points_worst': 'Titik Cekung (Terburuk)',
    'symmetry_worst': 'Simetri (Terburuk)',
    'fractal_dimension_worst': 'Dimensi Fraktal (Terburuk)',
}

# Balik mapping untuk keperluan tampilan
label_to_feature = {v: k for k, v in feature_labels.items()}

# ==================== INPUT FITUR ====================
st.sidebar.header("📋 Input Data Pasien")
st.sidebar.markdown("Masukkan nilai 30 fitur sel hasil FNA")

# Inisialisasi dictionary input
input_data = {}

# Kelompokkan fitur berdasarkan kategori
mean_features = [f for f in feature_names if f.endswith('_mean')]
se_features = [f for f in feature_names if f.endswith('_se')]
worst_features = [f for f in feature_names if f.endswith('_worst')]

with st.sidebar.expander("📊 Rata-rata (Mean)", expanded=True):
    for feat in mean_features:
        label = feature_labels[feat]
        default_val = float((mean_benign[feat] + mean_malignant[feat]) / 2)
        min_val = float(min(mean_benign[feat], mean_malignant[feat]) * 0.5)
        max_val = float(max(mean_benign[feat], mean_malignant[feat]) * 1.5)
        input_data[feat] = st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=0.001,
            format="%.3f"
        )

with st.sidebar.expander("📐 Std. Error (SE)", expanded=False):
    for feat in se_features:
        label = feature_labels[feat]
        default_val = float((mean_benign[feat] + mean_malignant[feat]) / 2)
        min_val = float(min(mean_benign[feat], mean_malignant[feat]) * 0.5)
        max_val = float(max(mean_benign[feat], mean_malignant[feat]) * 1.5)
        input_data[feat] = st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=0.001,
            format="%.3f"
        )

with st.sidebar.expander("🔥 Terburuk (Worst)", expanded=False):
    for feat in worst_features:
        label = feature_labels[feat]
        default_val = float((mean_benign[feat] + mean_malignant[feat]) / 2)
        min_val = float(min(mean_benign[feat], mean_malignant[feat]) * 0.5)
        max_val = float(max(mean_benign[feat], mean_malignant[feat]) * 1.5)
        input_data[feat] = st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=0.001,
            format="%.3f"
        )

# Tombol prediksi
predict_btn = st.sidebar.button("🔍 Prediksi", type="primary", use_container_width=True)

# ==================== MAIN AREA ====================
if predict_btn:
    # Buat DataFrame input
    input_df = pd.DataFrame([input_data])

    # Prediksi
    pred_class = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0]

    # === HASIL PREDIKSI ===
    st.subheader("📊 Hasil Prediksi")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if pred_class == 1:
            st.error("⚠️ **Malignant (Ganas)** — Segera konsultasikan ke dokter.")
        else:
            st.success("✅ **Benign (Jinak)** — Kemungkinan besar tidak berbahaya.")
    with col2:
        st.metric("Probabilitas Benign", f"{pred_prob[0]:.2%}")
    with col3:
        st.metric("Probabilitas Malignant", f"{pred_prob[1]:.2%}")

    # === PERBANDINGAN DENGAN HISTORIS ===
    st.markdown("---")
    st.subheader("📈 Perbandingan dengan Rata-rata Historis")
    top_features = ['radius_mean', 'texture_mean', 'perimeter_mean',
                    'area_mean', 'concave points_mean']
    comp_df = pd.DataFrame({
        'Fitur': [feature_labels[f] for f in top_features],
        'Pasien': [input_data[f] for f in top_features],
        'Rata-rata Benign': [mean_benign[f] for f in top_features],
        'Rata-rata Malignant': [mean_malignant[f] for f in top_features]
    })
    st.dataframe(comp_df.style.format({
        'Pasien': '{:.3f}',
        'Rata-rata Benign': '{:.3f}',
        'Rata-rata Malignant': '{:.3f}'
    }).background_gradient(cmap='RdYlGn_r', subset=['Pasien']))

    # ==================== INSIGHT 1: FEATURE IMPORTANCE ====================
with st.expander("🔍 Insight 1: Fitur Paling Berpengaruh", expanded=True):
    coef = model.named_steps['clf'].coef_[0]
    coef_df = pd.DataFrame({
        'Fitur': [feature_labels[f] for f in feature_names],
        'Fitur_Asli': feature_names,
        'Koefisien': coef
    }).sort_values('Koefisien', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = 10
    top_coef = coef_df.head(top_n)
    
    # === Warna GRADASI ===
    colors = []
    # Cari nilai maksimum absolut untuk normalisasi
    max_pos = top_coef[top_coef['Koefisien'] > 0]['Koefisien'].max() if any(top_coef['Koefisien'] > 0) else 1
    max_neg = abs(top_coef[top_coef['Koefisien'] < 0]['Koefisien'].min()) if any(top_coef['Koefisien'] < 0) else 1
    
    for v in top_coef['Koefisien']:
        if v > 0:
            intensity = min(v / max_pos, 1.0)
            # Merah gradasi: (1, 1-intensity, 1-intensity) -> semakin tinggi v, semakin merah
            colors.append((1.0, 1.0 - intensity * 0.8, 1.0 - intensity * 0.8))
        else:
            intensity = min(abs(v) / max_neg, 1.0)
            # Hijau gradasi: (1-intensity, 1, 1-intensity) -> semakin rendah v, semakin hijau
            colors.append((1.0 - intensity * 0.8, 1.0, 1.0 - intensity * 0.8))
    
    bars = ax.barh(top_coef['Fitur'], top_coef['Koefisien'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Koefisien (Logistic Regression)', fontsize=11)
    ax.set_title('Top 10 Fitur dengan Pengaruh Terbesar', fontsize=13, fontweight='bold')
    
    # Tambahkan nilai koefisien di ujung bar
    for bar, val in zip(bars, top_coef['Koefisien']):
        offset = 0.02 * (max_pos if val > 0 else max_neg)
        ha = 'left' if val > 0 else 'right'
        ax.text(val + (offset if val > 0 else -offset),
                bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', ha=ha, fontsize=9, fontweight='bold')
    
    st.pyplot(fig)
    
    st.markdown("""
    **Interpretasi:**
    - Warna **merah** → semakin tinggi nilai fitur, semakin **tinggi** risiko Malignant.
    - Warna **hijau** → semakin tinggi nilai fitur, semakin **rendah** risiko Malignant.
    - Semakin **gelap** warnanya, semakin besar pengaruh fitur tersebut.
    - Fitur paling berpengaruh: **Radius (Std. Error)** — peningkatan 1 unit meningkatkan odds Malignant ~14x.
    """)
    # ==================== INSIGHT 2: DECISION SPACE ====================
    with st.expander("📌 Insight 2: Ruang Keputusan (radius_mean vs texture_mean)", expanded=False):
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        # Titik rata-rata kelas
        ax2.scatter(mean_benign['radius_mean'], mean_benign['texture_mean'],
                    color='#2ecc71', s=120, label='Rata-rata Benign', alpha=0.8)
        ax2.scatter(mean_malignant['radius_mean'], mean_malignant['texture_mean'],
                    color='#e74c3c', s=120, label='Rata-rata Malignant', alpha=0.8)
        # Pasien baru
        ax2.scatter(input_data['radius_mean'], input_data['texture_mean'],
                    color='gold', s=250, marker='*', label='Pasien Baru', edgecolors='black')
        ax2.set_xlabel('Radius (Rata-rata)')
        ax2.set_ylabel('Tekstur (Rata-rata)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        st.markdown("""
        **Interpretasi:**
        - Titik hijau = rata-rata pasien jinak, titik merah = rata-rata pasien ganas.
        - Pasien baru (bintang) berada di area yang didominasi kelas tertentu.
        - Semakin jauh dari titik hijau dan mendekati merah, semakin tinggi risiko Malignant.
        """)

    # ==================== INSIGHT 3: PATIENT BENCHMARKING ====================
    with st.expander("📊 Insight 3: Perbandingan Pasien Baru vs Historis", expanded=False):
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
        ax3.set_xticklabels([feature_labels[f] for f in top_features], rotation=15, ha='right')
        ax3.set_ylabel('Nilai')
        ax3.legend()
        st.pyplot(fig3)
        st.markdown("""
        **Interpretasi:**
        - Bandingkan nilai pasien baru (oranye) dengan rata-rata Benign (hijau) dan Malignant (merah).
        - Jika batang oranye mendekati merah, maka indikasi Malignant lebih kuat.
        """)

    # ==================== INSIGHT 4: TEXTURE VS SMOOTHNESS ====================
    with st.expander("🧬 Insight 4: Tekstur vs Kehalusan Sel", expanded=False):
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        ax4.scatter(mean_benign['texture_mean'], mean_benign['smoothness_mean'],
                    color='#2ecc71', s=120, label='Rata-rata Benign', alpha=0.8)
        ax4.scatter(mean_malignant['texture_mean'], mean_malignant['smoothness_mean'],
                    color='#e74c3c', s=120, label='Rata-rata Malignant', alpha=0.8)
        ax4.scatter(input_data['texture_mean'], input_data['smoothness_mean'],
                    color='gold', s=250, marker='*', label='Pasien Baru', edgecolors='black')
        ax4.set_xlabel('Tekstur (Rata-rata)')
        ax4.set_ylabel('Kehalusan (Rata-rata)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        st.markdown("""
        **Interpretasi:**
        - Sel ganas cenderung memiliki tekstur lebih kasar (nilai tinggi) dan kehalusan lebih rendah.
        - Kombinasi tekstur tinggi + kehalusan rendah merupakan indikator kuat keganasan.
        """)

    # ==================== INSIGHT 5: KONVERGENSI PERFORM ====================
    with st.expander("📈 Insight 5: Konvergensi Performa Model", expanded=False):
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

        # Bar chart metrik
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        ax5.bar(metrics.keys(), metrics.values(), color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
        ax5.set_ylim(0.9, 1.0)
        ax5.set_ylabel('Skor')
        ax5.set_title('Performa Model pada Test Set')
        for i, v in enumerate(metrics.values()):
            ax5.text(i, v + 0.005, f"{v:.2%}", ha='center', fontsize=10)
        st.pyplot(fig5)

        st.markdown("""
        **Interpretasi:**
        - Semua metrik > 95%, dengan Recall 95.24% (prioritas utama).
        - ROC-AUC 99.64% menunjukkan kemampuan diskriminasi hampir sempurna.
        - Model robust dan siap digunakan sebagai alat bantu diagnosis.
        """)

    # ==================== RINGKASAN MODEL ====================
    st.markdown("---")
    st.subheader("📋 Ringkasan Model")
    st.markdown(f"""
    - **Model Terbaik**: GridSearchCV (Logistic Regression)
    - **Hyperparameter**: C=1, penalty=L1, solver=liblinear
    - **Recall** : {metrics['Recall']:.2%}
    - **F1 Score**: {metrics['F1 Score']:.2%}
    - **ROC-AUC** : {metrics['ROC-AUC']:.2%}
    - **Jumlah Fitur**: {len(feature_names)}
    """)

else:
    st.info("👈 Masukkan nilai fitur di sidebar, lalu klik tombol **'Prediksi'**.")

# ==================== FOOTER ====================
st.markdown("---")
st.caption("Dibuat untuk Final Project Machine Learning — Klasifikasi Kanker Payudara")
