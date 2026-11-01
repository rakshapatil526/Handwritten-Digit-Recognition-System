"""
Module 1: Home Page
- Project title and introduction
- Interactive display of digits 0-9
- Step-by-step explanation of how handwritten digit recognition works
- Feature highlights: Draw Digit, Instant Prediction, Prediction History
- "Start Drawing" button to trigger drawing workflow
"""

import os
import streamlit as st
from PIL import Image

def render_home_page():
    # Hero Title & Subtitle
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem 0 2rem 0;">
            <h1 style="font-size: 2.8rem; font-weight: 800; margin-bottom: 0.5rem; 
                       background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 50%, #EC4899 100%); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Handwritten Digit Recognition System
            </h1>
            <p style="font-size: 1.2rem; color: #64748B; max-width: 750px; margin: 0 auto; line-height: 1.6;">
                An intelligent deep learning web application powered by <b>TensorFlow</b> and <b>Streamlit</b>. 
                Draw any digit (0–9) on the interactive canvas or upload an image to receive real-time neural network predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Call to Action Button
    col_cta1, col_cta2, col_cta3 = st.columns([1, 1.5, 1])
    with col_cta2:
        if st.button("✨ Start Drawing a Digit Now →", type="primary", use_container_width=True):
            st.session_state["nav_selection"] = "✍️ Digit Recognition"
            st.rerun()

    st.markdown("<hr style='margin: 2rem 0; opacity: 0.2;'>", unsafe_allow_html=True)

    # Section 1: Display Digits 0–9
    st.subheader("🔢 MNIST Target Digits (0 to 9)")
    st.write(
        "The model is trained on tens of thousands of handwritten digit samples from the classic **MNIST dataset**. "
        "Here are sample digit templates recognized by the network:"
    )

    cols = st.columns(10)
    for digit in range(10):
        with cols[digit]:
            img_path = f"models/samples/digit_{digit}.png"
            if os.path.exists(img_path):
                img = Image.open(img_path).resize((70, 70), Image.Resampling.NEAREST)
                st.image(img, caption=f"Digit {digit}", use_container_width=True)
            else:
                st.markdown(
                    f"""
                    <div style="border: 2px dashed #94A3B8; border-radius: 8px; text-align: center; 
                                padding: 15px 0; font-size: 1.6rem; font-weight: bold; color: #3B82F6;">
                        {digit}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # Section 2: How It Works Pipeline
    st.subheader("🧠 How Handwritten Digit Recognition Works")
    st.write(
        "Transforming a freehand mouse stroke into an accurate digit prediction requires a robust pipeline "
        "of computer vision pre-processing and deep convolutional neural networks:"
    )

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown(
            """
            <div style="background-color: rgba(59, 130, 246, 0.08); border-left: 4px solid #3B82F6; 
                        padding: 1rem; border-radius: 6px; height: 100%;">
                <h4 style="color: #2563EB; margin-top: 0;">1. Freehand Canvas Capture</h4>
                <p style="font-size: 0.95rem; color: #475569; margin-bottom: 0;">
                    You sketch a digit using your mouse or touchscreen. The application captures the drawing as high-resolution RGBA pixel matrix data.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_p2:
        st.markdown(
            """
            <div style="background-color: rgba(139, 92, 246, 0.08); border-left: 4px solid #8B5CF6; 
                        padding: 1rem; border-radius: 6px; height: 100%;">
                <h4 style="color: #7C3AED; margin-top: 0;">2. MNIST Standardization</h4>
                <p style="font-size: 0.95rem; color: #475569; margin-bottom: 0;">
                    The stroke is isolated via bounding box, scaled proportionally into a 20×20 area, centered via <b>Center of Mass</b> on a 28×28 canvas, and normalized to [0, 1].
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_p3:
        st.markdown(
            """
            <div style="background-color: rgba(236, 72, 153, 0.08); border-left: 4px solid #EC4899; 
                        padding: 1rem; border-radius: 6px; height: 100%;">
                <h4 style="color: #DB2777; margin-top: 0;">3. CNN Feature Extraction</h4>
                <p style="font-size: 0.95rem; color: #475569; margin-bottom: 0;">
                    Convolutional and pooling layers detect curves, loops, and intersections, outputting a <b>Softmax probability distribution</b> across all digits (0–9).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # Section 3: Feature Modules Overview
    st.subheader("🚀 System Features")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.markdown(
            """
            ### ✍️ Draw Digit
            - Interactive touch & mouse drawing canvas
            - Pen stroke size and eraser adjustments
            - Instant canvas clearing and one-click redo
            - Image upload option (.png, .jpg, .jpeg)
            """
        )
    with f_col2:
        st.markdown(
            """
            ### ⚡ Instant Prediction
            - Real-time CNN inference
            - Top predicted digit with confidence badge
            - Full 10-class probability bar chart
            - 28×28 preprocessed input preview
            """
        )
    with f_col3:
        st.markdown(
            """
            ### 📊 Prediction History
            - Session history tracking for all predictions
            - Thumbnail previews and confidence scores
            - MNIST dataset architecture breakdown
            - Interactive training loss and accuracy charts
            """
        )

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # Bottom CTA
    st.info("💡 **Ready to test the neural network?** Click the button below or choose **Digit Recognition** in the sidebar!")
    if st.button("Go to Digit Recognition Page", use_container_width=True):
        st.session_state["nav_selection"] = "✍️ Digit Recognition"
        st.rerun()
