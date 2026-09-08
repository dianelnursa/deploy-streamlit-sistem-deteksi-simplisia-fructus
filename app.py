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
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f9f7;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .header-box {
        background: linear-gradient(135deg, #1b5e20, #388e3c);
        padding: 35px 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.10);
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

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #1b5e20;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .info-box {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #388e3c;
        box-shadow: 0 3px 15px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
    }

    .result-box {
        background-color: white;
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        margin-top: 20px;
        border-top: 5px solid #388e3c;
    }

    .result-label {
        color: #666;
        font-size: 15px;
        margin-bottom: 8px;
    }

    .result-name {
        color: #1b5e20;
        font-size: 27px;
        font-weight: 700;
    }

    .confidence {
        font-size: 18px;
        margin-top: 12px;
        color: #444;
    }

    .indication-box {
        background-color: white;
        padding: 22px;
        border-radius: 15px;
        margin-top: 15px;
        box-shadow: 0 3px 15px rgba(0, 0, 0, 0.05);
    }

    .footer {
        text-align: center;
        color: #777;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
    }

    [data-testid="stFileUploader"] {
        background-color: white;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 3px 15px rgba(0, 0, 0, 0.05);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FUNGSI PREDIKSI
# =========================================================

def import_and_predict(image_data, model):

    # Ukuran input mengikuti model yang digunakan saat training
    size = (128, 128)

    # Resize dan crop gambar
    image = ImageOps.fit(
        image_data,
        size,
        method=Image.Resampling.LANCZOS
    )

    # Pastikan gambar RGB
    image = image.convert("RGB")

    # Ubah menjadi array
    image = np.asarray(image)

    # Normalisasi pixel 0-255 menjadi 0-1
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
    <div class="header-box">

        <div class="header-title">
            🌿 Deteksi Simplisia Fructus
        </div>

        <div class="header-subtitle">
            Sistem Klasifikasi Simplisia Fructus
            Berbasis Convolutional Neural Network (CNN)
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

        Aplikasi ini digunakan untuk membantu mengidentifikasi
        jenis <b>simplisia fructus</b> berdasarkan citra atau
        gambar yang diunggah oleh pengguna.

        <br><br>

        Sistem saat ini dapat mengenali
        <b>7 jenis simplisia fructus</b> menggunakan model
        Convolutional Neural Network (CNN).

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DAFTAR KELAS
# =========================================================

with st.expander("🌱 Lihat 7 jenis simplisia yang dapat dideteksi"):

    st.markdown(
        """
        **1. Amomi Fructus**  
        Kapulaga

        **2. Capsici Frutescentis Fructus**  
        Cabai Rawit

        **3. Cumini Fructus**  
        Jinten

        **4. Piper Retrofractum Fructus**  
        Cabai Jawa

        **5. Piperis Albi Fructus**  
        Lada Putih

        **6. Piperis Nigri Fructus**  
        Lada Hitam

        **7. Tamarindus indica Fructus**  
        Asam Jawa
        """
    )


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = load_model()

except Exception as e:

    st.error(
        "Model tidak dapat dimuat."
    )

    st.warning(
        "Pastikan file 'Xception-fructus-98.19.h5' "
        "sudah tersedia di repository GitHub."
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
    "Silakan pilih gambar simplisia fructus "
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
        "👆 Belum ada gambar yang diunggah. "
        "Silakan upload gambar untuk memulai deteksi."
    )


# =========================================================
# JIKA ADA GAMBAR
# =========================================================

else:

    # Buka gambar
    image = Image.open(file)

    # Pastikan RGB
    image = image.convert("RGB")

    # =====================================================
    # PREVIEW GAMBAR
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

    with st.spinner("Sedang menganalisis gambar..."):

        prediction = import_and_predict(
            image,
            model
        )


    # =====================================================
    # AMBIL HASIL PREDIKSI
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
    # TAMPILKAN HASIL
    # =====================================================

    st.markdown(
        f"""
        <div class="result-box">

            <div class="result-label">
                Hasil Klasifikasi
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

            st.write(
                f"{class_names[index]} — "
                f"{probability * 100:.2f}%"
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
        '<div class="indication-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        "**Pemanfaatan secara umum:**"
    )

    for item in indications[predicted_class]:

        st.markdown(
            f"- {item}"
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        <b>Deteksi Simplisia Fructus</b><br>

        Sistem klasifikasi berbasis
        Convolutional Neural Network (CNN)

    </div>
    """,
    unsafe_allow_html=True
)
