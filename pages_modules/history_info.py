"""
Module 3: Prediction History & Model Information
- Previous digit predictions (preview image, predicted number, confidence, timestamp)
- Model accuracy and evaluation metrics
- MNIST dataset information
- CNN architecture information
- Training accuracy and loss graphs (interactive Plotly charts)
- Confusion matrix and per-class performance metrics
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from utils.predictor import load_training_metadata

def render_history_and_info_page():
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="margin-bottom: 0.2rem; font-weight: 700;">📊 Prediction History & Model Information</h2>
            <p style="color: #64748B;">Inspect past session predictions, explore the CNN architecture, and review MNIST training performance metrics.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    metadata = load_training_metadata()
    history = st.session_state.get("prediction_history", [])

    # Top-level tabs for clean organization
    tab_history, tab_model_info = st.tabs(["📜 Prediction History", "🧠 Model & Dataset Architecture"])

    # ==========================================================
    # TAB 1: PREDICTION HISTORY
    # ==========================================================
    with tab_history:
        st.subheader("📜 Session Prediction History")

        if not history:
            st.info("ℹ️ No predictions recorded in this session yet. Draw a digit in the **Digit Recognition** page to see your history here!")
            if st.button("✍️ Go to Digit Recognition", type="primary"):
                st.session_state["nav_selection"] = "✍️ Digit Recognition"
                st.rerun()
        else:
            # Stats & Clear button
            h_stat1, h_stat2, h_stat3, h_stat4 = st.columns([1, 1, 1, 1])
            with h_stat1:
                st.metric("Total Predictions", len(history))
            with h_stat2:
                avg_conf = sum(item["confidence"] for item in history) / len(history)
                st.metric("Average Confidence", f"{avg_conf:.1f}%")
            with h_stat3:
                digits_seen = [item["predicted_digit"] for item in history]
                most_common = max(set(digits_seen), key=digits_seen.count)
                st.metric("Most Frequent Digit", str(most_common))
            with h_stat4:
                st.write("")
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state["prediction_history"] = []
                    st.rerun()

            st.markdown("<hr style='margin: 1.2rem 0; opacity: 0.2;'>", unsafe_allow_html=True)

            # Display cards of past predictions (most recent first)
            for idx, item in enumerate(reversed(history)):
                card_col1, card_col2, card_col3, card_col4, card_col5 = st.columns([1, 1.2, 1.5, 1.8, 1.5])
                
                with card_col1:
                    preview = item.get("preview_image")
                    if preview:
                        st.image(preview.resize((65, 65), Image.Resampling.NEAREST), caption="Digit Input", width=65)
                    else:
                        st.write("🖼️")

                with card_col2:
                    st.markdown(
                        f"""
                        <div style="font-size: 2.2rem; font-weight: 800; color: #2563EB; line-height: 1.1;">
                            {item['predicted_digit']}
                        </div>
                        <div style="font-size: 0.8rem; color: #64748B;">Predicted</div>
                        """,
                        unsafe_allow_html=True
                    )

                with card_col3:
                    conf = item["confidence"]
                    c_color = "#10B981" if conf >= 85 else ("#F59E0B" if conf >= 60 else "#EF4444")
                    st.markdown(
                        f"""
                        <div style="font-weight: 700; color: {c_color}; font-size: 1.1rem; padding-top: 0.3rem;">
                            {conf:.1f}%
                        </div>
                        <div style="font-size: 0.8rem; color: #64748B;">Confidence</div>
                        """,
                        unsafe_allow_html=True
                    )

                with card_col4:
                    st.markdown(
                        f"""
                        <div style="font-size: 0.95rem; font-weight: 500; padding-top: 0.3rem;">
                            {item.get('source', 'Canvas')}
                        </div>
                        <div style="font-size: 0.8rem; color: #64748B;">Source</div>
                        """,
                        unsafe_allow_html=True
                    )

                with card_col5:
                    st.markdown(
                        f"""
                        <div style="font-size: 0.85rem; color: #64748B; padding-top: 0.3rem;">
                            ⏱️ {item.get('timestamp', 'N/A')}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("<hr style='margin: 0.8rem 0; opacity: 0.1;'>", unsafe_allow_html=True)

    # ==========================================================
    # TAB 2: MODEL & DATASET INFORMATION
    # ==========================================================
    with tab_model_info:
        if metadata is None:
            st.warning("⚠️ Training metadata not yet available. Please wait for model training to finish.")
            return

        # Top Model Performance Metrics
        st.subheader("🎯 Model Performance Metrics")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Test Accuracy", f"{metadata.get('test_accuracy', 0)}%", delta="MNIST Test Set")
        with m_col2:
            st.metric("Test Loss", f"{metadata.get('test_loss', 0):.4f}")
        with m_col3:
            st.metric("Total Parameters", f"{metadata.get('total_parameters', 0):,}")
        with m_col4:
            st.metric("Trained Epochs", f"{metadata.get('epochs', 0)} epochs")

        st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.2;'>", unsafe_allow_html=True)

        # Dataset & CNN Architecture Cards
        d_col, a_col = st.columns([1, 1], gap="large")

        with d_col:
            st.subheader("📚 MNIST Dataset Information")
            d_info = metadata.get("dataset_info", {})
            st.markdown(
                f"""
                - **Full Name:** {d_info.get('name', 'MNIST')}
                - **Authors:** {d_info.get('authors', 'Yann LeCun et al.')}
                - **Training Samples:** {d_info.get('training_samples', 60000):,} images
                - **Test Samples:** {d_info.get('test_samples', 10000):,} images
                - **Image Dimensions:** {d_info.get('image_dimensions', '28 x 28 pixels (Grayscale)')}
                - **Classes:** 10 digits (`0, 1, 2, 3, 4, 5, 6, 7, 8, 9`)
                - **Normalization:** Raw uint8 pixel values [0, 255] scaled to [0.0, 1.0]
                """
            )

        with a_col:
            st.subheader("🏗️ CNN Architecture Overview")
            st.markdown(
                """
                - **Input Layer:** Shape `(28, 28, 1)` Grayscale Tensor
                - **Block 1:** Conv2D (32 filters, 3×3, ReLU) → BatchNorm → Conv2D (32 filters) → BatchNorm → MaxPool2D (2×2) → Dropout (0.25)
                - **Block 2:** Conv2D (64 filters, 3×3, ReLU) → BatchNorm → Conv2D (64 filters) → BatchNorm → MaxPool2D (2×2) → Dropout (0.25)
                - **Dense Block:** Flatten → Dense (128 units, ReLU) → BatchNorm → Dropout (0.50)
                - **Output Layer:** Dense (10 units, Softmax activation)
                - **Optimizer:** Adam (lr=0.001) | **Loss:** Sparse Categorical Crossentropy
                """
            )

        st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.2;'>", unsafe_allow_html=True)

        # Training Accuracy and Loss Graphs
        st.subheader("📈 Training Accuracy & Loss History")
        
        hist = metadata.get("history", {})
        epochs_range = list(range(1, len(hist.get("accuracy", [])) + 1))

        g_col1, g_col2 = st.columns(2)

        with g_col1:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=epochs_range, y=hist.get("accuracy", []),
                mode="lines+markers", name="Training Accuracy",
                line=dict(color="#3B82F6", width=2.5),
                marker=dict(size=7)
            ))
            fig_acc.add_trace(go.Scatter(
                x=epochs_range, y=hist.get("val_accuracy", []),
                mode="lines+markers", name="Validation Accuracy",
                line=dict(color="#10B981", width=2.5, dash="dash"),
                marker=dict(size=7)
            ))
            fig_acc.update_layout(
                title="Model Accuracy vs. Epochs",
                xaxis_title="Epoch",
                yaxis_title="Accuracy",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=320,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_acc, use_container_width=True, key="fig_train_val_acc")

        with g_col2:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=epochs_range, y=hist.get("loss", []),
                mode="lines+markers", name="Training Loss",
                line=dict(color="#EF4444", width=2.5),
                marker=dict(size=7)
            ))
            fig_loss.add_trace(go.Scatter(
                x=epochs_range, y=hist.get("val_loss", []),
                mode="lines+markers", name="Validation Loss",
                line=dict(color="#F59E0B", width=2.5, dash="dash"),
                marker=dict(size=7)
            ))
            fig_loss.update_layout(
                title="Model Loss vs. Epochs",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=320,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_loss, use_container_width=True, key="fig_train_val_loss")

        st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.2;'>", unsafe_allow_html=True)

        # Per-Class Accuracy & Confusion Matrix
        c_col1, c_col2 = st.columns([1.1, 1], gap="large")

        with c_col1:
            st.subheader("📊 Per-Digit Classification Accuracy")
            per_class = metadata.get("per_class_accuracy", {})
            if per_class:
                df_per_class = pd.DataFrame([
                    {"Digit": f"Digit {d}", "Accuracy (%)": v["accuracy"], "Total Test Samples": v["total"]}
                    for d, v in per_class.items()
                ])
                fig_bar = px.bar(
                    df_per_class,
                    x="Digit",
                    y="Accuracy (%)",
                    text="Accuracy (%)",
                    color="Accuracy (%)",
                    color_continuous_scale="Blues",
                    range_y=[95, 100]
                )
                fig_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                fig_bar.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True, key="fig_per_class_bar")

        with c_col2:
            st.subheader("🔍 Confusion Matrix Heatmap")
            cm = metadata.get("confusion_matrix", [])
            if cm:
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted Digit", y="Actual Digit", color="Count"),
                    x=[str(i) for i in range(10)],
                    y=[str(i) for i in range(10)],
                    color_continuous_scale="Viridis",
                    aspect="auto"
                )
                fig_cm.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_cm, use_container_width=True, key="fig_confusion_matrix")

        # Layer Details Table
        with st.expander("🔬 View Detailed Layer-by-Layer Parameters"):
            layers_data = metadata.get("layers", [])
            if layers_data:
                df_layers = pd.DataFrame(layers_data)
                df_layers.columns = ["Layer Name", "Layer Type", "Output Shape", "Parameters"]
                st.dataframe(df_layers, use_container_width=True)
