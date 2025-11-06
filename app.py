import os
import numpy as np
import cv2
import gdown
import tensorflow as tf
import streamlit as st

st.set_page_config(
    page_title="🐛 Pest Classifier CNN",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🐜 Pest Classifier CNN")
st.markdown("""
Aplikasi ini mendeteksi jenis serangga dari gambar yang diunggah.
Menggunakan model CNN untuk klasifikasi 12 kelas serangga umum.
""")

MODEL_PATH = "pest_classifier_cnn.h5"
url = "https://drive.google.com/uc?id=19-zwiD61CRi-a6Nkf6XRcQvt-X1oWgPo"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Mengunduh model dari Google Drive (~340MB)..."):
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True, use_cookies=True)

file_size = os.path.getsize(MODEL_PATH)
st.write(f"📦 Ukuran file model: {file_size / (1024*1024):.2f} MB")
if file_size < 100_000:
    st.error("❌ File model terlalu kecil — kemungkinan bukan file .h5 yang valid.")
    st.stop()

with st.spinner("🔄 Memuat model CNN..."):
    model = tf.keras.models.load_model(MODEL_PATH)

classes = [
    'ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig',
    'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil'
]

def full_preprocessing_pipeline(image_array, size=(256, 256)):
    original_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    compressed = cv2.resize(original_rgb, size)
    gray = cv2.cvtColor(compressed, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    blurred = cv2.GaussianBlur(compressed, (3, 3), 0)
    img_yuv = cv2.cvtColor(blurred, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    normalized = enhanced / 255.0
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

uploaded_file = st.file_uploader("📤 Upload gambar serangga", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    data = full_preprocessing_pipeline(img)

    with st.expander("🔍 Tahapan Lengkap Preprocessing", expanded=True):
        st.image(data["original"], caption="1️⃣ Citra Asli (RGB)", width=300)
        st.image(data["compressed"], caption="2️⃣ Resize (256x256)", width=300)

        cols = st.columns(2)
        cols[0].image(data["gray"], caption="3️⃣ Grayscale", width=250)
        cols[1].image(data["binary"], caption="4️⃣ Binerisasi (Otsu)", width=250)

        cols2 = st.columns(2)
        cols2[0].image(data["blurred"], caption="5️⃣ Gaussian Blur", width=250)
        cols2[1].image(data["enhanced"], caption="6️⃣ Histogram Equalization", width=250)

        st.image(data["normalized"], caption="7️⃣ Normalisasi (0–1)", width=300)

    with st.spinner("🤖 Memprediksi jenis serangga..."):
        preds = model.predict(data["input_tensor"])
        pred_idx = np.argmax(preds)
        confidence = np.max(preds) * 100
        predicted_class = classes[pred_idx]

    st.subheader("📊 Hasil Klasifikasi")
    st.success(f"🐞 Prediksi: **{predicted_class.upper()}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

    prob_table = {classes[i]: float(preds[0][i]) for i in range(len(classes))}
    st.table(prob_table)
