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
        border-left: 5px solid #
