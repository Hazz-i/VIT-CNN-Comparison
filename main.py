import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Lettuce Disease Classifier", layout="centered")

st.title("🥬 Klasifikasi Penyakit Daun Selada")
st.write("Unggah foto daun selada untuk mendeteksi apakah sehat atau terkena penyakit.")

# --- LOAD MODEL (DI-CACHE AGAR CEPAT) ---
@st.cache_resource
def load_prediction_model(model_name):
    # Tentukan path folder berdasarkan pilihan
    path = "cnn_model" if model_name == "CNN Model" else "vit_model"
    
    # Memasukkan SavedModel ke dalam Sequential sebagai TFSMLayer
    # Berdasarkan log notebook, endpoint-nya adalah 'serve'
    model = tf.keras.Sequential([
        tf.keras.layers.TFSMLayer(path, call_endpoint="serve")
    ])
    return model

# --- SIDEBAR: PILIH MODEL ---
st.sidebar.header("Pengaturan Model")
selected_model_name = st.sidebar.selectbox(
    "Pilih Arsitektur Model:",
    ("CNN Model", "ViT Model")
)

# Load model yang dipilih
model = load_prediction_model(selected_model_name)

# --- DAFTAR KELAS ---
# Sesuai dengan urutan di notebook: Bacterial, Fungal, Healthy
class_names = ['Bacterial', 'Fungal', 'Healthy']

# --- UI UNTUK UPLOAD GAMBAR ---
uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_container_width=True)
    
    st.write("---")
    st.write(f"**Prediksi menggunakan: {selected_model_name}**")

    # --- PREPROCESSING ---
    # 1. Resize ke 224x224 sesuai IMG_SIZE di notebook
    img = image.resize((224, 224))
    # 2. Konversi ke array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # 3. Normalisasi (rescale 1./255)
    img_array = img_array / 255.0
    # 4. Tambah dimensi batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # --- PREDIKSI ---
    # Jalankan prediksi satu kali saja
    preds = model.predict(img_array)
    
    # Penanganan output Keras 3 (bisa berupa dict atau numpy array)
    if isinstance(preds, dict):
        # Jika output berupa dictionary, ambil value pertama dan batch pertama
        prediction = list(preds.values())[0][0]
    else:
        # Jika output sudah berupa numpy array, langsung ambil batch pertama
        prediction = preds[0]

    # result_index: Indeks kelas dengan probabilitas tertinggi
    result_index = np.argmax(prediction)
    # confidence: Nilai probabilitas tertinggi dalam persen
    confidence = np.max(prediction) * 100

    # --- TAMPILKAN HASIL ---
    st.subheader(f"Hasil Prediksi: **{class_names[result_index]}**")
    st.progress(int(confidence))
    st.write(f"Tingkat Keyakinan: **{confidence:.2f}%**")