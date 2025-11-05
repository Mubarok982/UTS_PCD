import os
import gdown
import tensorflow as tf
import streamlit as st

MODEL_PATH = "pest_classifier_cnn.h5"
DRIVE_URL = "https://drive.google.com/uc?id=19-zwiD61CRi-a6Nkf6XRcQvt-X1oWgPo"

# === Download model otomatis dari Google Drive ===
if not os.path.exists(MODEL_PATH):
    st.info("📥 Mengunduh model dari Google Drive... (~340MB)")
    try:
        # gunakan use_cookies=True agar bisa bypass konfirmasi ukuran file besar
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=True, fuzzy=True, use_cookies=True)
    except Exception as e:
        st.error(f"❌ Gagal mengunduh model: {e}")
        st.stop()

# === Verifikasi ukuran file ===
if not os.path.exists(MODEL_PATH):
    st.error("❌ File model tidak ditemukan setelah unduhan.")
    st.stop()

file_size = os.path.getsize(MODEL_PATH)
st.write(f"📦 Ukuran file model: {file_size / (1024*1024):.2f} MB")

if file_size < 100_000:
    st.error("❌ File model terlalu kecil — kemungkinan bukan file .h5 yang valid (mungkin HTML).")
    st.stop()

# === Load model CNN ===
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model berhasil dimuat dan siap digunakan!")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()


#Label Kelas
classes = [
    'ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig',
    'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil'
]

#Fungsi Preprocessing Lengkap
def full_preprocessing_pipeline(image_array, size=(256, 256)):
    # Citra Asli
    original_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # Pemampatan (Resize)
    compressed = cv2.resize(original_rgb, size)

    # Grayscale
    gray = cv2.cvtColor(compressed, cv2.COLOR_RGB2GRAY)

    # Binerisasi (Threshold)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Filtering (Gaussian Blur)
    blurred = cv2.GaussianBlur(compressed, (3, 3), 0)

    # Enhancement (Histogram Equalization)
    img_yuv = cv2.cvtColor(blurred, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    # Normalisasi (0–1)
    normalized = enhanced / 255.0

    # Siap Masuk Model CNN
    input_tensor = np.expand_dims(normalized, axis=0)

    return {
        "original": original_rgb,
        "compressed": compressed,
        "gray": gray,
        "binary": binary,
        "blurred": blurred,
        "enhanced": enhanced,
        "normalized": normalized,
        "input_tensor": input_tensor
    }

#Upload dan Proses Gambar
uploaded_file = st.file_uploader("📤 Upload gambar serangga", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Jalankan semua tahap preprocessing
    data = full_preprocessing_pipeline(img)

    # Tampilkan hasil preprocessing
    st.subheader("🔍 Tahapan Lengkap Preprocessing")
    st.image(data["original"], caption="1️⃣ Citra Asli (RGB)", use_column_width=True)
    st.image(data["compressed"], caption="2️⃣ Citra Setelah Resize (256x256)", use_column_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(data["gray"], caption="3️⃣ Grayscale", use_column_width=True)
    with col2:
        st.image(data["binary"], caption="4️⃣ Binerisasi (Otsu)", use_column_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.image(data["blurred"], caption="5️⃣ Gaussian Blur", use_column_width=True)
    with col4:
        st.image(data["enhanced"], caption="6️⃣ Enhancement (Histogram Equalization)", use_column_width=True)

    st.image(data["normalized"], caption="7️⃣ Normalisasi (0–1)", use_column_width=True)

    # Prediksi dengan model CNN
    preds = model.predict(data["input_tensor"])
    pred_idx = np.argmax(preds)
    confidence = np.max(preds) * 100
    predicted_class = classes[pred_idx]

    # Hasil Prediksi
    st.subheader("📊 Hasil Klasifikasi")
    st.success(f"Hasil Prediksi: **{predicted_class.upper()}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Probabilitas semua kelas
    st.write("🔢 Probabilitas masing-masing kelas:")
    prob_table = {classes[i]: float(preds[0][i]) for i in range(len(classes))}
    st.table(prob_table)
