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
    layout="centered"
)


# =========================================================
# CSS / TAMPILAN
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f8f5;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .header {
        background: linear-gradient(135deg, #1b5e20, #43a047);
        padding: 35px 25px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
    }

    .header h1 {
        font-size: 34px;
        margin-bottom: 8px;
    }

    .header p {
        font-size: 16px;
        margin: 0;
    }

    .info-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #43a047;
        margin: 15px 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.06);
    }

    .result-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        margin-top: 20px;
        border-top: 5px solid #43a047;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }

    .result-title {
        color: #777;
        font-size: 15px;
    }

    .result-name {
        color: #1b5e20;
        font-size: 27px;
        font-weight: bold;
        margin: 10px 0;
    }

    .confidence {
        font-size: 17px;
        color: #444;
    }

    .section-title {
        color: #1b5e20;
        font-size: 23px;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .footer {
        text-align: center;
        color: #777;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FUNGSI PREDIKSI
# =========================================================

def import_and_predict(image_data, model):

    # Ukuran input mengikuti model asli
    size = (128, 128)

    # Resize gambar
    image = ImageOps.fit(
        image_data,
        size,
        method=Image.Resampling.LANCZOS
    )

    # Pastikan RGB
    image = image.convert("RGB")

    # Konversi ke NumPy
    image = np.asarray(image)

    # Normalisasi
    image = image.astype(np.float32) / 255.0

    # Tambahkan dimensi batch
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
    <div class="header">

        🌿 Deteksi Simplisia Fructus

        
            Sistem Klasifikasi Simplisia Fructus
            Berbasis Convolutional Neural Network (CNN)
        

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# INFORMASI
# =========================================================

st.markdown(
    '<div class="section-title">Tentang Aplikasi</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">

    Aplikasi ini digunakan untuk membantu mendeteksi jenis
    <b>simplisia fructus</b> berdasarkan gambar yang diunggah.

    Sistem dapat mengenali <b>7 jenis simplisia fructus</b>
    berdasarkan model klasifikasi berbasis
    <b>Convolutional Neural Network (CNN)</b>.

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DAFTAR KELAS
# =========================================================

with st.expander("🌱 Lihat jenis simplisia yang dapat dideteksi"):

    st.markdown(
        """
        **1. Amomi Fructus** — Kapulaga

        **2. Cumini Fructus** — Jinten

        **3. Piperis Albi Fructus** — Lada Putih

        **4. Piperis Nigri Fructus** — Lada Hitam

        **5. Piper Retrofractum Fructus** — Cabai Jawa

        **6. Tamarindus indica Fructus** — Asam Jawa

        **7. Capsici Frutescentis Fructus** — Cabai Rawit
        """
    )


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = load_model()

except Exception as e:

    st.error("❌ Model tidak dapat dimuat.")

    st.write(
        "Pastikan file berikut tersedia di repository:"
    )

    st.code(
        "Xception-fructus-98.19.h5"
    )

    st.stop()


# =========================================================
# UPLOAD GAMBAR
# =========================================================

st.markdown(
    '<div class="section-title">📷 Upload Gambar</div>',
    unsafe_allow_html=True
)

st.write(
    "Silakan upload gambar simplisia fructus "
    "dengan format JPG, JPEG, atau PNG."
)

file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# BELUM ADA GAMBAR
# =========================================================

if file is None:

    st.info(
        "👆 Silakan upload gambar untuk memulai proses deteksi."
    )


# =========================================================
# JIKA GAMBAR SUDAH DIUPLOAD
# =========================================================

else:

    # Buka gambar
    image = Image.open(file)

    # Konversi RGB
    image = image.convert("RGB")


    # =====================================================
    # TAMPILKAN GAMBAR
    # =====================================================

    st.markdown(
        '<div class="section-title">🖼️ Gambar yang Diunggah</div>',
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption="Gambar simplisia fructus",
        use_container_width=True
    )


    # =====================================================
    # PREDIKSI
    # =====================================================

    st.markdown(
        '<div class="section-title">🔍 Hasil Deteksi</div>',
        unsafe_allow_html=True
    )

    with st.spinner(
        "Sedang menganalisis gambar..."
    ):

        prediction = import_and_predict(
            image,
            model
        )


    # =====================================================
    # HASIL PREDIKSI
    # =====================================================

    predicted_class = int(
        np.argmax(prediction)
    )

    confidence = float(
        np.max(prediction)
    ) * 100


    # =====================================================
    # NAMA KELAS
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


    # =====================================================
    # INFORMASI PEMANFAATAN
    # =====================================================

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
            "Memiliki rasa dan aroma khas."
        ],

        4: [
            "Digunakan sebagai penyedap alami dalam masakan.",
            "Memberikan rasa khas pada berbagai makanan.",
            "Digunakan sebagai bahan rempah."
        ],

        5: [
            "Digunakan sebagai penyedap alami dalam masakan.",
            "Secara tradisional digunakan untuk membantu pencernaan.",
            "Digunakan sebagai rempah dalam berbagai hidangan."
        ],

        6: [
            "Digunakan sebagai bumbu dalam berbagai hidangan.",
            "Memberikan rasa asam khas pada makanan dan minuman.",
            "Dimanfaatkan sebagai bahan pangan dan rempah."
        ]
    }


    # =====================================================
    # CARD HASIL
    # =====================================================

    st.markdown(
        f"""
        <div class="result-card">

            <div class="result-title">
                HASIL TERDETEKSI
            </div>

            <div class="result-name">
                {class_names[predicted_class]}
            </div>

            <div class="confidence">
                Tingkat keyakinan:
                <b>{confidence:.2f}%</b>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # PROBABILITAS
    # =====================================================

    with st.expander("📊 Lihat probabilitas setiap kelas"):

        for index, probability in enumerate(
            prediction[0]
        ):

            percentage = float(
                probability
            ) * 100

            st.write(
                f"**{class_names[index]}** "
                f"— {percentage:.2f}%"
            )

            st.progress(
                float(probability)
            )


    # =====================================================
    # INFORMASI SIMPLISIA
    # =====================================================

    st.markdown(
        '<div class="section-title">🌿 Informasi Simplisia</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">
        <b>Pemanfaatan secara umum:</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    for item in indications[predicted_class]:

        st.markdown(
            f"- {item}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        Deteksi Simplisia Fructus

        Sistem klasifikasi berbasis
        Convolutional Neural Network (CNN)

    </div>
    """,
    unsafe_allow_html=True
)
