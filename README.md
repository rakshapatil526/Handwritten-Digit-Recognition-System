# ✍️ Handwritten Digit Recognition System

An end-to-end, interactive Deep Learning web application built using **Python**, **Streamlit**, and **TensorFlow/Keras (Convolutional Neural Network - CNN)** to recognize handwritten digits (0–9) in real-time.

---

## 🌟 Key Features & 3 Modules

### 1. 🏠 Home Page
- **Hero Title & Overview:** Clean introduction to handwritten digit recognition and deep learning concepts.
- **MNIST Digits 0–9 Showcase:** Interactive gallery displaying sample handwritten digit templates from the MNIST dataset.
- **Workflow Pipeline Explanation:** Visual, step-by-step breakdown of how freehand drawing is processed into a neural network prediction:
  1. *Freehand Canvas Capture*
  2. *MNIST Standardization (Bounding Box, 20×20 Aspect-Ratio Preserving Scale, Center of Mass Alignment, and Normalization)*
  3. *CNN Feature Extraction & Softmax Classification*
- **Feature Highlights:** Direct links and descriptions for Draw Digit, Instant Prediction, and History.
- **"Start Drawing" CTA Button:** Instantly redirects to the interactive canvas module.

### 2. ✍️ Digit Recognition
- **Interactive Drawing Canvas:** Powered by `streamlit-drawable-canvas`, write any digit using a mouse, stylus, or touchscreen.
- **Drawing Controls:**
  - Pen and Eraser tool switching.
  - Stroke width adjustment slider (12px to 36px).
  - Dark Canvas (Black background with white pen) or Light Canvas with automatic MNIST color inversion.
- **Clear Canvas & Try Again Buttons:** One-click reset to draw a fresh digit.
- **Instant Prediction Output:**
  - Top predicted digit highlighted with an eye-catching confidence badge.
  - Interactive **Plotly** probability distribution bar chart across all 10 digits (0 through 9).
  - Top 3 alternative predictions with percentage scores.
  - Visual 28×28 Preprocessed Input preview (showing what the neural network actually sees).
- **Image Upload Support:** Upload custom digit photos (`.png`, `.jpg`, `.jpeg`) alongside the drawing canvas.

### 3. 📊 Prediction History & Model Information
- **Session Prediction History:**
  - Card view of all past predictions with timestamp, input source (Canvas or Upload), predicted digit badge, and confidence level.
  - Session summary statistics: Total predictions, Average confidence score, Most frequent digit.
  - One-click *Clear History* button.
- **Model Performance & Architecture:**
  - Test accuracy (~99.1%) and test loss.
  - Total and trainable parameters breakdown.
  - MNIST dataset details (60,000 training, 10,000 testing, 28×28 grayscale, normalized pixel values, Yann LeCun benchmark).
  - Full CNN architecture specification (Conv2D, Batch Normalization, Max Pooling, Dropout, Dense, Softmax).
  - Interactive **Plotly Training Curves**: Epoch vs. Accuracy (Train/Val) & Epoch vs. Loss (Train/Val).
  - Interactive **Confusion Matrix Heatmap** across all 10 digits.
  - **Per-Digit Classification Accuracy** chart.

---

## 🏗️ Neural Network Architecture

```
Input Layer (28 x 28 x 1 Grayscale Image)
│
├── Conv2D (32 filters, 3x3, ReLU) + BatchNormalization
├── Conv2D (32 filters, 3x3, ReLU) + BatchNormalization
├── MaxPooling2D (2x2) + Dropout (0.25)
│
├── Conv2D (64 filters, 3x3, ReLU) + BatchNormalization
├── Conv2D (64 filters, 3x3, ReLU) + BatchNormalization
├── MaxPooling2D (2x2) + Dropout (0.25)
│
├── Flatten
├── Dense (128 units, ReLU) + BatchNormalization + Dropout (0.50)
└── Output Dense (10 units, Softmax) -> Probabilities for Digits 0–9
```

---

## 🚀 Installation & Running

### 1. Clone or Open the Project
```bash
cd "Handwritten Digit Recognition System"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train the Model
The pre-trained CNN model and metadata are saved in `models/`. To retrain from scratch:
```bash
python train_model.py
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
