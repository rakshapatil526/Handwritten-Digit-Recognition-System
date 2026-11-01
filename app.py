"""
Handwritten Digit Recognition System - Main Application
Framework: Streamlit + TensorFlow/Keras CNN
Author: AI & ML Projects
"""

import streamlit as st

# Apply compatibility patch before any component load
from utils.patch import apply_streamlit_patch
apply_streamlit_patch()

from pages_modules.home import render_home_page
from pages_modules.recognition import render_recognition_page
from pages_modules.history_info import render_history_and_info_page
from utils.predictor import load_digit_model

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition System",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Clean tab buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 1rem;
        padding-top: 8px;
        padding-bottom: 8px;
    }
    
    /* Card borders */
    div.block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize navigation state
if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "🏠 Home Page"

# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding-bottom: 1rem;">
            <div style="font-size: 2.8rem; margin-bottom: 0.3rem;">🧠✍️</div>
            <h2 style="margin: 0; font-size: 1.35rem; font-weight: 800; color: #1E293B;">Digit AI Recognizer</h2>
            <p style="font-size: 0.85rem; color: #64748B; margin-top: 0.2rem;">CNN Digit Recognition System</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Navigation")
    pages = [
        "🏠 Home Page",
        "✍️ Digit Recognition",
        "📊 Prediction History & Model Info"
    ]
    
    # Map index
    current_idx = 0
    if st.session_state["nav_selection"] in pages:
        current_idx = pages.index(st.session_state["nav_selection"])

    selected_page = st.radio(
        "Select Module:",
        options=pages,
        index=current_idx,
        label_visibility="collapsed"
    )
    st.session_state["nav_selection"] = selected_page

    st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.2;'>", unsafe_allow_html=True)

    # Sidebar System Status
    st.markdown("### System Status")
    model = load_digit_model()
    if model is not None:
        st.success("🟢 CNN Model Loaded & Ready")
    else:
        st.warning("🟡 Model Loading / Training...")

    # Quick Stats
    history_count = len(st.session_state.get("prediction_history", []))
    st.write(f"📝 **Session Predictions:** `{history_count}`")
    st.write("📐 **Input Resolution:** `28 × 28 px`")
    st.write("🎯 **Classes:** `10 (Digits 0–9)`")

    st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
    st.caption("Built with Python, Streamlit, TensorFlow/Keras & NumPy.")

# Routing
if st.session_state["nav_selection"] == "🏠 Home Page":
    render_home_page()
elif st.session_state["nav_selection"] == "✍️ Digit Recognition":
    render_recognition_page()
elif st.session_state["nav_selection"] == "📊 Prediction History & Model Info":
    render_history_and_info_page()
