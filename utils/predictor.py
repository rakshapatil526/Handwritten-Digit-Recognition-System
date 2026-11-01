"""
Model loading and digit inference utility.
Provides cached loading and prediction functions.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st

MODEL_PATH = "models/mnist_cnn.keras"
METADATA_PATH = "models/training_metadata.json"

@st.cache_resource
def load_digit_model():
    """
    Load the trained Keras CNN model. Caches the model instance in memory.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@st.cache_data
def load_training_metadata():
    """
    Load metadata and training history saved during training.
    """
    if not os.path.exists(METADATA_PATH):
        return None
    try:
        with open(METADATA_PATH, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

def predict_digit(model, tensor_input):
    """
    Runs inference on the preprocessed tensor input (shape: 1, 28, 28, 1).
    
    Returns:
    - predicted_digit: int (0-9)
    - confidence: float (0.0 to 100.0)
    - probabilities: list of floats for digits 0-9
    - top_3: list of dicts with digit and confidence
    """
    if model is None or tensor_input is None:
        return None
    
    # Model inference
    preds = model.predict(tensor_input, verbose=0)[0]
    probs = [float(p) for p in preds]
    
    pred_digit = int(np.argmax(probs))
    conf = float(probs[pred_digit]) * 100.0
    
    # Sort top 3
    indexed_probs = list(enumerate(probs))
    indexed_probs.sort(key=lambda x: x[1], reverse=True)
    top_3 = [{"digit": d, "confidence": round(p * 100.0, 2)} for d, p in indexed_probs[:3]]
    
    return {
        "predicted_digit": pred_digit,
        "confidence": round(conf, 2),
        "probabilities": [round(p * 100.0, 2) for p in probs],
        "top_3": top_3
    }
