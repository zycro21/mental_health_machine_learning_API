# 🧠 Mental Health Machine Learning API

API sederhana berbasis **FastAPI** untuk memprediksi tingkat depresi berdasarkan data gangguan mental lainnya menggunakan model machine learning (CatBoost).

Proyek ini dirancang untuk menjadi layanan prediksi yang cepat, ringan, dan bisa diintegrasikan dengan sistem backend utama (misal aplikasi klinik atau rumah sakit).

---

## 🚀 Cara Menjalankan Aplikasi (Sekali Copy Jalan)

Pastikan kamu sudah menginstal **Python 3.8+** dan sudah memiliki file model `best_model_catboost.pkl`.

```bash
# 1. Clone repository ini
git clone https://github.com/yourusername/mental_health_machine_learning_API.git
cd mental_health_machine_learning_API

# 2. Buat virtual environment
python -m venv venv
source venv/bin/activate           # Linux/macOS
.\venv\Scripts\activate            # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan aplikasi
uvicorn main:app --reload

---

## 🧠 Tentang Model Machine Learning

Model yang digunakan dalam proyek ini adalah **CatBoost Regressor**, salah satu algoritma Gradient Boosting yang dikembangkan oleh Yandex. CatBoost dirancang untuk bekerja sangat baik dengan data kategori, tetapi juga sangat efisien untuk data numerik.

### 🔍 Kenapa Memilih CatBoost?

- Performa tinggi tanpa perlu preprocessing rumit.
- Tidak membutuhkan encoding manual untuk data kategorikal.
- Mendukung CPU dan GPU training.
- Tidak terlalu sensitif terhadap hyperparameter seperti model boosting lainnya (XGBoost/LightGBM).

---

## 📊 Data dan Fitur yang Digunakan

Model dilatih menggunakan dataset open-source terkait **statistik gangguan mental** dari berbagai negara dan tahun, yang mencakup:

- `schizophrenia_share`: Persentase penderita skizofrenia.
- `anxiety_share`: Persentase penderita gangguan kecemasan.
- `bipolar_share`: Persentase penderita bipolar.
- `eating_disorder_share`: Persentase penderita eating disorder.
- `DALYs`: Disability Adjusted Life Years untuk gangguan depresi.
- `suicide_rate`: Angka bunuh diri per 100.000 jiwa.
- `depression_dalys`: DALYs khusus untuk depresi.
- `schizophrenia_dalys`: DALYs khusus untuk skizofrenia.
- `bipolar_dalys`: DALYs khusus untuk bipolar.
- `eating_dalys`: DALYs khusus untuk eating disorder.
- `anxiety_dalys`: DALYs khusus untuk anxiety disorder.

Target prediksi dari model ini adalah **skor depresi kuantitatif** (regresi), yang kemudian dikategorikan menjadi:

- `Rendah` (≤ 1.0)
- `Sedang` (> 1.0 dan ≤ 3.0)
- `Tinggi` (> 3.0)

---

## 🧪 Pelatihan Model (Training)

Proses pelatihan dilakukan dengan langkah-langkah umum berikut:

1. **Data preprocessing**: Menggabungkan berbagai fitur numerik dari dataset mental health.
2. **Split data**: Memisahkan data menjadi training dan validation set.
3. **Training**: Menggunakan `CatBoostRegressor` dari library `catboost` dengan parameter default atau hasil tuning ringan.
4. **Evaluasi**: Menggunakan metrik seperti RMSE atau MAE untuk mengukur performa.
5. **Model saving**: Setelah performa optimal tercapai, model disimpan dengan `joblib.dump()` menjadi `best_model_catboost.pkl`.