"""
Module 2: Digit Recognition
- Drawing canvas to write a digit using mouse or touchscreen
- Pen and Eraser tools with stroke width slider
- Clear Canvas button and Try Again button
- Real-time / Instant Predict Digit button
- Probability distribution bar chart (Plotly)
- Visual preprocessed preview (28x28 centered MNIST standard)
- Prediction history logging into st.session_state
"""

import time
import io
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Ensure patch is applied before importing canvas
from utils.patch import apply_streamlit_patch
apply_streamlit_patch()
from streamlit_drawable_canvas import st_canvas

from utils.preprocessor import preprocess_canvas_image, preprocess_uploaded_image
from utils.predictor import load_digit_model, predict_digit

def render_recognition_page():
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="margin-bottom: 0.2rem; font-weight: 700;">✍️ Handwritten Digit Recognition</h2>
            <p style="color: #64748B;">Draw a single digit (0–9) on the canvas below, or upload an image, and let the CNN model predict it!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load model
    model = load_digit_model()
    if model is None:
        st.warning("⚠️ The CNN model is currently training or not yet saved. Please wait a moment...")
        return

    # Initialize session state for canvas key and history
    if "canvas_key_id" not in st.session_state:
        st.session_state["canvas_key_id"] = 0
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []
    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None

    # Tabs for Drawing vs Uploading
    tab_draw, tab_upload = st.tabs(["🖌️ Draw on Canvas", "📤 Upload Image"])

    # ==========================================================
    # TAB 1: DRAW ON CANVAS
    # ==========================================================
    with tab_draw:
        col_canvas, col_results = st.columns([1.1, 1], gap="large")

        with col_canvas:
            st.markdown("#### 🎨 Canvas Controls")
            
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])
            with ctrl_col1:
                drawing_mode = st.radio(
                    "Tool:",
                    options=["freedraw", "eraser"],
                    format_func=lambda x: "🖊️ Pen" if x == "freedraw" else "🧹 Eraser",
                    horizontal=True
                )
            with ctrl_col2:
                stroke_width = st.slider("Stroke Width:", min_value=12, max_value=36, value=22, step=2)
            with ctrl_col3:
                canvas_theme = st.selectbox("Canvas Background:", ["Black (Recommended)", "White"])
            
            # Setup canvas configuration
            if canvas_theme.startswith("Black"):
                bg_color = "#000000"
                stroke_color = "#FFFFFF" if drawing_mode == "freedraw" else "#000000"
                invert_needed = False
            else:
                bg_color = "#FFFFFF"
                stroke_color = "#000000" if drawing_mode == "freedraw" else "#FFFFFF"
                invert_needed = True

            st.caption("💡 *Tip: Draw a bold, centered digit filling roughly 60–80% of the canvas height.*")

            # The Drawable Canvas
            canvas_key = f"digit_canvas_{st.session_state['canvas_key_id']}"
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                height=300,
                width=300,
                drawing_mode="freedraw",
                key=canvas_key,
                return_image_data=True
            )

            # Action Buttons
            btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1, 1])
            with btn_col1:
                predict_clicked = st.button("🚀 Predict Digit", type="primary", use_container_width=True)
            with btn_col2:
                if st.button("🧹 Clear Canvas", use_container_width=True):
                    st.session_state["canvas_key_id"] += 1
                    st.session_state["last_prediction"] = None
                    st.rerun()
            with btn_col3:
                if st.button("🔄 Try Again", use_container_width=True):
                    st.session_state["canvas_key_id"] += 1
                    st.session_state["last_prediction"] = None
                    st.rerun()

            # Handle prediction trigger
            if predict_clicked:
                if canvas_result is not None and canvas_result.image_data is not None:
                    tensor, preview_pil, is_blank = preprocess_canvas_image(
                        canvas_result.image_data, 
                        invert_colors=invert_needed
                    )
                    if is_blank:
                        st.warning("⚠️ The canvas appears to be empty or the drawing was too faint. Please draw a clear digit!")
                    else:
                        pred_result = predict_digit(model, tensor)
                        if pred_result:
                            # Save to session history
                            history_item = {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "source": "Canvas Drawing",
                                "predicted_digit": pred_result["predicted_digit"],
                                "confidence": pred_result["confidence"],
                                "preview_image": preview_pil,
                                "probabilities": pred_result["probabilities"]
                            }
                            st.session_state["prediction_history"].append(history_item)
                            st.session_state["last_prediction"] = {
                                "result": pred_result,
                                "preview": preview_pil,
                                "source": "Canvas Drawing"
                            }
                            st.rerun()
                else:
                    st.warning("⚠️ The canvas is empty. Please draw a digit before clicking Predict!")

        with col_results:
            st.markdown("#### 🎯 Prediction Results")
            _display_prediction_card(st.session_state.get("last_prediction"), key_suffix="draw")

    # ==========================================================
    # TAB 2: UPLOAD IMAGE
    # ==========================================================
    with tab_upload:
        u_col1, u_col2 = st.columns([1, 1], gap="large")
        with u_col1:
            st.markdown("#### 📤 Upload Digit Image")
            uploaded_file = st.file_uploader(
                "Choose an image of a handwritten digit (.png, .jpg, .jpeg):",
                type=["png", "jpg", "jpeg"]
            )
            if uploaded_file is not None:
                img_display = Image.open(uploaded_file)
                st.image(img_display, caption="Uploaded Original Image", width=220)
                
                if st.button("🚀 Predict Uploaded Digit", type="primary", use_container_width=True):
                    tensor, preview_pil, is_blank = preprocess_uploaded_image(uploaded_file)
                    if is_blank:
                        st.warning("⚠️ Could not detect a digit in the uploaded image. Please try another image.")
                    else:
                        pred_result = predict_digit(model, tensor)
                        if pred_result:
                            history_item = {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "source": "Uploaded File",
                                "predicted_digit": pred_result["predicted_digit"],
                                "confidence": pred_result["confidence"],
                                "preview_image": preview_pil,
                                "probabilities": pred_result["probabilities"]
                            }
                            st.session_state["prediction_history"].append(history_item)
                            st.session_state["last_prediction"] = {
                                "result": pred_result,
                                "preview": preview_pil,
                                "source": "Uploaded File"
                            }
                            st.rerun()

        with u_col2:
            st.markdown("#### 🎯 Prediction Results")
            _display_prediction_card(st.session_state.get("last_prediction"), key_suffix="upload")


def _display_prediction_card(pred_data, key_suffix="draw"):
    """
    Renders the prediction badges, confidence scores, probability bar chart,
    and 28x28 preprocessed model view.
    """
    if not pred_data or "result" not in pred_data:
        st.markdown(
            """
            <div style="border: 2px dashed #CBD5E1; border-radius: 12px; padding: 3rem 1.5rem; text-align: center; color: #94A3B8;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔮</div>
                <h4 style="color: #64748B; margin-bottom: 0.5rem;">No Prediction Yet</h4>
                <p style="font-size: 0.9rem; margin: 0;">Draw a digit on the canvas and click <b>Predict Digit</b> to see the neural network output.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    result = pred_data["result"]
    preview = pred_data["preview"]
    pred_digit = result["predicted_digit"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]
    top_3 = result["top_3"]

    # Top Prediction Card
    conf_color = "#10B981" if confidence >= 85 else ("#F59E0B" if confidence >= 60 else "#EF4444")
    
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%); 
                    border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 12px; padding: 1.5rem; text-align: center; margin-bottom: 1.2rem;">
            <div style="font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 600;">
                Predicted Digit
            </div>
            <div style="font-size: 4.8rem; font-weight: 900; line-height: 1; color: #1E293B; margin: 0.4rem 0;">
                {pred_digit}
            </div>
            <div style="display: inline-block; background-color: {conf_color}20; color: {conf_color}; 
                        font-weight: 700; font-size: 1.05rem; padding: 4px 16px; border-radius: 20px; border: 1px solid {conf_color}60;">
                {confidence:.1f}% Confidence
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Side-by-side Top 3 & Preprocessed 28x28 image
    sub_col1, sub_col2 = st.columns([1.2, 1])
    with sub_col1:
        st.markdown("**Top Alternative Predictions:**")
        for item in top_3:
            d = item["digit"]
            c = item["confidence"]
            st.write(f"Digit **{d}**: {c:.1f}%")

    with sub_col2:
        st.markdown("**Model Input (28×28):**")
        if preview:
            st.image(preview.resize((100, 100), Image.Resampling.NEAREST), caption="Centered MNIST Tensor", width=100)

    # Plotly Probability Bar Chart
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("**Class Probability Distribution (0–9):**")

    df_probs = pd.DataFrame({
        "Digit": [str(i) for i in range(10)],
        "Probability (%)": probabilities
    })

    # Color top prediction differently
    colors = ["#3B82F6" if i != pred_digit else "#10B981" for i in range(10)]

    fig = go.Figure(
        data=[
            go.Bar(
                x=df_probs["Digit"],
                y=df_probs["Probability (%)"],
                marker_color=colors,
                text=[f"{p:.1f}%" if p > 3 else "" for p in probabilities],
                textposition="auto",
                hovertemplate="Digit %{x}: %{y:.2f}%<extra></extra>"
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        yaxis=dict(range=[0, 105], title="Confidence (%)"),
        xaxis=dict(title="Digit Class"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True, key=f"prob_chart_{key_suffix}")
