import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Deteksi Simplisia Fructus",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Background */
    .stApp {
        background-color: #f7f9f7;
    }

    /* Main container */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .header-box {
        background: linear-gradient(135deg, #1b5e20, #388e3c);
        padding: 35px 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.10);
    }

    .header-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .header-subtitle {
        font-size: 16px;
        opacity: 0.92;
    }

    /* Section */
    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #1b5e20;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    /* Info box */
    .info-box {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #388e3c;
        box-shadow: 0 3px 15px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    /* Result */
    .result-box {
        background-color: white;
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin-top: 20px;
        border-top: 5px solid #388e3c;
    }

    .result-label {
        color: #666;
        font-size: 15px;
        margin-bottom: 5px;
    }

    .result-name {
        color: #1b5e20;
        font-size: 27px;
        font-weight: 700;
    }

    /* Indication */
    .indication-box {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 15px;
        margin-top: 15px;
        box-shadow: 0 3px 15px rgba(0,0,0,0.05);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
    }

    /* Upload */
    [data-testid="stFileUploader"] {
        background-color: white;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 3px 15px rgba(0,0,0,0.05);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FUNGSI PREDIKSI
# =========================================================

def import_and_predict(image_data, model):

    # Ukuran input mengikuti model lama
    size = (128, 128)

    # Resize image
    image = ImageOps.fit(
        image_data,
        size,
        method=Image.Resampling.LANCZOS
    )

    # Pastikan RGB
    image = image.convert("RGB")

    # Convert ke numpy
    image = np.asarray(image)

    # Normalisasi
    image = image.astype(np.float32) / 255.0

    # Tambahkan batch dimension
    img_reshape = image[np.newaxis, ...]

    # Prediksi
    prediction = model.predict(
        img_reshape,
        verbose=0
    )

    return prediction


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "Xception-fructus-98.19.h5",
        compile=False
    )

    return model


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="header-box">
        <div class="header-title">
            🌿 Deteksi Simplisia Fructus
        </div>
        <div class="header-subtitle">
            Sistem klasifikasi jenis simplisia fructus berbasis
            Convolutional Neural Network (CNN)
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# INFORMASI APLIKASI
# =========================================================

st.markdown(
    '<div class="section-title">Tentang Aplikasi</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
        Aplikasi ini digunakan untuk membantu mengidentifikasi jenis
        simplisia fructus berdasarkan citra atau gambar yang diunggah.
        Sistem saat ini dapat mengenali <b>7 jenis simplisia fructus</b>.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DAFTAR KELAS
# =========================================================

with st.expander("Lihat 7 jenis simplisia yang dapat dideteksi"):

    st.markdown(
        """
        1. **Amomi Fructus** — Kapulaga  
        2. **Capsici Frutescentis Fructus** — Cabai Rawit  
        3. **Cumini Fructus** — Jinten  
        4. **Piper Retrofractum Fructus** — Cabai Jawa  
        5. **Piperis Albi Fructus** — Lada Putih  
        6. **Piperis Nigri Fructus** — Lada Hitam  
        7. **Tamarindus indica Fructus** — Asam Jawa
        """
    )


# =========================================================
# LOAD MODEL
# =========================================================

try:
    model = load_model()

except Exception as e:

    st.error(
        "Model tidak dapat dimuat. Pastikan file "
        "`Xception-fructus-98.19.h5` tersedia di repository."
    )

    st.stop()


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">Upload Gambar</div>',
    unsafe_allow_html=True
)

st.write(
    "Pilih gambar simplisia fructus dalam format JPG atau PNG."
)

file = st.file_uploader(
    "Silakan upload gambar simplisia",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# JIKA BELUM ADA GAMBAR
# =========================================================

if file is None:

    st.info(
        "Belum ada gambar yang diunggah. "
        "Silakan pilih gambar untuk memulai proses deteksi."
    )


# =========================================================
# JIKA ADA GAMBAR
# =========================================================

else:

    image = Image.open(file)

    st.image(
        image,
        caption="Gambar yang diunggah",
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Hasil Deteksi</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Sedang menganalisis gambar..."):

        prediction = import_and_predict(
            image,
            model
        )

    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100


    # =====================================================
    # DATA KELAS
    # =====================================================

    class_names = {
        0: "Amomi Fructus / Kapulaga",
        1: "Capsici Frutescentis Fructus / Cabai Rawit",
        2: "Cumini Fructus / Jinten",
        3: "Piper Retrofractum Fructus / Cabai Jawa",
        4: "Piperis Albi Fructus / Lada Putih",
        5: "Piperis Nigri Fructus / Lada Hitam",
        6: "Tamarindus indica Fructus / Asam Jawa"
    }


    indications = {

        0: [
            "Membantu pencernaan.",
            "Digunakan sebagai penyedap dan pemberi aroma pada makanan dan minuman.",
            "Memiliki aroma khas yang banyak dimanfaatkan dalam pengolahan bahan pangan."
        ],

        1: [
            "Digunakan sebagai bumbu dalam berbagai masakan.",
            "Memberikan rasa pedas pada makanan.",
            "Mengandung senyawa yang menjadi karakteristik cabai."
        ],

        2: [
            "Digunakan sebagai penyedap alami dalam berbagai hidangan.",
            "Secara tradisional digunakan untuk membantu pencernaan.",
            "Mengandung senyawa yang memiliki aktivitas antioksidan."
        ],

        3: [
            "Digunakan sebagai bahan dalam berbagai ramuan tradisional.",
            "Dimanfaatkan sebagai bumbu dan bahan herbal.",
            "Memiliki
