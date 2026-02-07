# 🥬 Klasifikasi Penyakit Daun Selada

Aplikasi web berbasis Streamlit untuk mendeteksi penyakit pada daun selada menggunakan model Deep Learning (CNN dan Vision Transformer).

## 📋 Fitur

- Upload gambar daun selada untuk klasifikasi
- Pilihan 2 arsitektur model: **CNN** dan **Vision Transformer (ViT)**
- Klasifikasi 3 kelas: `Bacterial`, `Fungal`, `Healthy`
- Menampilkan tingkat keyakinan prediksi

## 🛠️ Instalasi

### Prasyarat

- Python 3.10 atau lebih baru
- [uv](https://github.com/astral-sh/uv) (package installer)

### Langkah-langkah Setup

1. **Clone repository**

   ```bash
   git clone <repository-url>
   cd vit-cnn-klasifikasi-jamur
   ```

2. **Buat virtual environment**

   ```bash
   python -m venv .venv
   ```

3. **Aktifkan virtual environment**
   - **Windows (PowerShell):**

     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```

   - **Windows (CMD):**

     ```cmd
     .\.venv\Scripts\activate.bat
     ```

   - **Linux/macOS:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies menggunakan uv pip**

   ```bash
   uv pip install -r requirements.txt
   ```

## 🚀 Menjalankan Aplikasi

Setelah instalasi selesai, jalankan aplikasi dengan:

```bash
streamlit run main.py
```

Aplikasi akan terbuka di browser pada alamat `http://localhost:8501`

## 📁 Struktur Proyek

```
vit-cnn-klasifikasi-jamur/
├── main.py              # Aplikasi utama Streamlit
├── requirements.txt     # Daftar dependencies
├── README.md            # Dokumentasi
├── cnn_model/           # Model CNN (SavedModel format)
│   ├── saved_model.pb
│   ├── fingerprint.pb
│   ├── assets/
│   └── variables/
└── vit_model/           # Model ViT (SavedModel format)
    ├── saved_model.pb
    ├── fingerprint.pb
    ├── assets/
    └── variables/
```

## 📦 Dependencies

- `tensorflow==2.20.0`
- `streamlit==1.54.0`
- `pillow==12.1.0`

## 📝 Cara Penggunaan

1. Buka aplikasi di browser
2. Pilih model yang ingin digunakan (CNN atau ViT) di sidebar
3. Upload gambar daun selada (format: JPG, JPEG, atau PNG)
4. Lihat hasil prediksi dan tingkat keyakinan