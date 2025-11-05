import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

st.set_page_config(page_title="Klasifikasi Serangga Pertanian", page_icon="🐞", layout="centered")
st.title("🦋 Klasifikasi Serangga Pertanian Menggunakan CNN")
st.write("""
Aplikasi ini menampilkan **seluruh tahapan preprocessing citra digital** sebelum dilakukan klasifikasi menggunakan model CNN.
""")

MODEL_PATH = "pest_classifier_cnn.h5"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model tidak ditemukan! Pastikan file `pest_classifier_model.h5` ada di folder ini.")
    st.stop()

model = load_model(MODEL_PATH)
st.success("✅ Model berhasil dimuat!")

classes = [
    'ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig',
    'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil'
]

def full_preprocessing_pipeline(image_array, size=(256, 256)):
    # --- Citra Asli ---
    original_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # --- Pemampatan (Resize) ---
    compressed = cv2.resize(original_rgb, size)

    # --- Grayscale ---
    gray = cv2.cvtColor(compressed, cv2.COLOR_RGB2GRAY)

    # --- Binerisasi (Threshold) ---
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # --- Filtering (Gaussian Blur) ---
    blurred = cv2.GaussianBlur(compressed, (3, 3), 0)

    # --- Enhancement (Histogram Equalization) ---
    gray_eq = cv2.equalizeHist(gray)

    # --- Normalisasi (0–1) ---
    normalized = blurred / 255.0

    # --- Siap Masuk Model CNN ---
    input_tensor = np.expand_dims(normalized, axis=0)

    return {
        "original": original_rgb,
        "compressed": compressed,
        "gray": gray,
        "binary": binary,
        "blurred": blurred,
        "enhanced": gray_eq,
        "normalized": normalized,
        "input_tensor": input_tensor
    }

uploaded_file = st.file_uploader("📤 Upload gambar serangga", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Jalankan semua tahap preprocessing
    data = full_preprocessing_pipeline(img)

    st.subheader("🔍 Tahapan Lengkap Preprocessing")

    st.image(data["original"], caption="1️⃣ Citra Asli (RGB)", use_column_width=True)
    st.image(data["compressed"], caption="2️⃣ Citra Setelah Pemampatan (Resize 256x256)", use_column_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(data["gray"], caption="3️⃣ Citra Grayscale", use_column_width=True)
    with col2:
        st.image(data["binary"], caption="4️⃣ Citra Biner (Threshold Otsu)", use_column_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.image(data["blurred"], caption="5️⃣ Filtering (Gaussian Blur)", use_column_width=True)
    with col4:
        st.image(data["enhanced"], caption="6️⃣ Enhancement (Histogram Equalization)", use_column_width=True)

    st.image(data["normalized"], caption="7️⃣ Normalisasi (Nilai Piksel 0–1)", use_column_width=True)

    preds = model.predict(data["input_tensor"])
    pred_idx = np.argmax(preds)
    confidence = np.max(preds) * 100
    predicted_class = classes[pred_idx]

    st.subheader("📊 Hasil Klasifikasi")
    st.success(f"Hasil Prediksi: **{predicted_class.upper()}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Tampilkan probabilitas semua kelas
    st.write("🔢 Probabilitas masing-masing kelas:")
    prob_table = {classes[i]: float(preds[0][i]) for i in range(len(classes))}
    st.table(prob_table)