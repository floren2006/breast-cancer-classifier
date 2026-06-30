# CDSS Klasifikasi Kanker Payudara — Streamlit App

Aplikasi Streamlit untuk mendukung skripsi klasifikasi kanker payudara
(Logistic Regression, Breast Cancer Wisconsin Dataset).

## ⚠️ Perubahan Penting di Paket Ini

Versi sebelumnya punya masalah: `build_artifacts.py` membangun ulang
`data.csv` dari dataset bawaan scikit-learn dan menjalankan Optuna hanya
**50 trial**, sedangkan notebook skripsi (`breast_cancer_classification_FINAL.ipynb`)
memakai `data.csv` asli kamu dan Optuna **100 trial**. Akibatnya angka yang
tampil di Streamlit berisiko tidak identik dengan angka di laporan.

Paket ini sudah diperbaiki:
- `model/data.csv` sekarang adalah **file data.csv asli** yang kamu upload
  (569 baris, sama persis dengan yang dipakai notebook), bukan hasil generate ulang.
- `build_artifacts.py` sekarang membaca `model/data.csv` ini langsung (tidak
  lagi memanggil `sklearn.datasets.load_breast_cancer`), dan Optuna dijalankan
  **100 trial**, sama seperti notebook.

## ❗ Langkah yang HARUS kamu lakukan sebelum app.py bisa dijalankan

Folder `model/` di paket ini **belum berisi file `.pkl`** (`best_model_logreg.pkl`,
`benchmark_stats.pkl`, `results.pkl`). Ini disengaja — model harus dilatih
ulang di environment-mu sendiri (di sandbox pembuatan paket ini, koneksi
internet dimatikan sehingga `optuna` tidak bisa diinstal untuk
menjalankan training secara real). Langkahnya:

```bash
# 1. (opsional) buat virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. install dependency aplikasi
pip install -r requirements.txt

# 3. install optuna (hanya dibutuhkan untuk membangun model, bukan saat app jalan)
pip install optuna

# 4. jalankan script build — proses ini melatih Default, GridSearchCV,
#    RandomizedSearchCV, dan Optuna (100 trial). Pada dataset 569 baris ini
#    biasanya selesai dalam 1-3 menit.
python build_artifacts.py

# 5. setelah selesai, model/best_model_logreg.pkl, model/benchmark_stats.pkl,
#    dan model/results.pkl akan otomatis terbentuk

# 6. jalankan aplikasi
streamlit run app.py
```

Setelah langkah ini, angka di Dashboard / Insight Analytics / Model Evaluation
akan identik (deterministik, karena `random_state=42` dipakai konsisten di
setiap tahap) dengan yang dihasilkan notebook, **selama kamu memakai
`model/data.csv` yang sama** dan tidak mengubah parameter di `build_artifacts.py`.

## Struktur Folder

```
streamlit_app/
├── app.py                 # Aplikasi utama (semua halaman)
├── build_artifacts.py     # Script untuk melatih model & membuat artefak (jalankan dulu!)
├── requirements.txt
├── README.md
└── model/
    ├── data.csv               # Dataset ASLI (sama dengan yang dipakai notebook)
    ├── best_model_logreg.pkl  # (dibuat oleh build_artifacts.py)
    ├── benchmark_stats.pkl    # (dibuat oleh build_artifacts.py)
    └── results.pkl            # (dibuat oleh build_artifacts.py)
```

## Halaman Aplikasi

| Halaman | Deskripsi |
|---|---|
| 🏠 Dashboard | Ringkasan penelitian dan performa model |
| 📊 Dataset Explorer | Eksplorasi data, statistik deskriptif, missing value, korelasi |
| 📈 Insight Analytics | 5 insight: Feature Importance, PR Curve, Confusion Matrix, Feature Distribution, Hyperparameter Convergence |
| 🧬 Patient Prediction | Input data pasien baru → prediksi Benign/Malignant |
| 🔍 Prediction Explanation | Penjelasan kontribusi fitur terhadap prediksi terakhir |
| 📋 Model Evaluation | ROC Curve, PR Curve, Confusion Matrix, Classification Report per metode tuning |
| ℹ️ About Model | Penjelasan dataset, algoritma, alur penelitian, glosarium fitur medis |

## Melatih Ulang Model (jika dataset/strategi berubah)

```bash
python build_artifacts.py
```
Script ini menjalankan ulang seluruh pipeline (Default, GridSearchCV,
RandomizedSearchCV, Optuna 100 trial) dan menyimpan ulang semua file di `model/`.

## Troubleshooting

- **FileNotFoundError best_model_logreg.pkl saat `streamlit run app.py`** → kamu
  belum menjalankan `python build_artifacts.py`. Ini wajib dilakukan sekali
  sebelum app.py bisa jalan (lihat bagian "Langkah yang HARUS kamu lakukan" di atas).
- **ModuleNotFoundError: optuna** → `pip install optuna` (cukup untuk tahap build, tidak perlu untuk `streamlit run app.py`).
- **ModuleNotFoundError lain** → pastikan `pip install -r requirements.txt` sudah dijalankan di environment yang aktif.
- **Versi scikit-learn berbeda menimbulkan warning saat load `.pkl`** → jalankan ulang `build_artifacts.py` di environment kamu agar model dilatih ulang dengan versi library yang sama dengan yang dipakai saat `streamlit run`.
