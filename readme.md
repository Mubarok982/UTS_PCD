# 🦋 Klasifikasi Serangga Pertanian Menggunakan CNN

Aplikasi ini dibuat menggunakan **Streamlit**, **OpenCV**, dan **TensorFlow**,  
untuk mengklasifikasikan berbagai jenis serangga pertanian berdasarkan citra digital.

---

## 🚀 Fitur Utama
- Menampilkan seluruh **tahapan preprocessing citra**:
  1. Konversi warna (RGB)
  2. Resize (256x256)
  3. Grayscale
  4. Threshold (Otsu)
  5. Gaussian Blur
  6. Histogram Equalization (Enhancement)
  7. Normalisasi (0–1)
- Menggunakan **model CNN** untuk klasifikasi 12 kelas serangga:
  - `ants`, `bees`, `beetle`, `catterpillar`, `earthworms`, `earwig`,
    `grasshopper`, `moth`, `slug`, `snail`, `wasp`, `weevil`.
- Menampilkan hasil prediksi dan confidence score.
- Tampilan interaktif berbasis **Streamlit Web App**.

---

## 🧠 Model CNN

Model CNN digunakan untuk klasifikasi citra serangga, namun karena ukurannya **>100MB**,  
file model **tidak disimpan di GitHub** agar repositori tetap ringan.

👉 **Download model di sini:**  
[📥 Download pest_classifier_cnn.h5 (Google Drive)](https://drive.google.com/file/d/19-zwiD61CRi-a6Nkf6XRcQvt-X1oWgPo/view?usp=sharing)

> 💡 Pastikan kamu menempatkan file `pest_classifier_cnn.h5` di direktori yang sama dengan `app.py`.

Atau biarkan aplikasi otomatis mengunduh model jika belum ada,  
karena sudah dilengkapi fitur download otomatis menggunakan `gdown`.

---

## ⚙️ Cara Menjalankan Aplikasi

### 1. Clone repository
```bash
git clone https://github.com/Mubarok982/UTS_PCD.git
cd UTS_PCD
```

### 2. Install dependensi
```bash
pip install -r requirements.txt
pastikan sudah menggunakan python versi 3.9+
```

### 3.Jalankan aplikasi Streamlit
```bash
streamlit run app.py
```