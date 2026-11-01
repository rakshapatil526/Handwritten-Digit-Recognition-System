"""
Training script for Handwritten Digit Recognition System using TensorFlow/Keras CNN on MNIST.
Saves:
- Trained model: models/mnist_cnn.keras
- Training history and evaluation metadata: models/training_metadata.json
- Sample digit images (0-9): models/samples/
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

def build_model():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        
        # Conv block 1
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Conv block 2
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Dense block
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    return model

def main():
    print("TensorFlow Version:", tf.__version__)
    os.makedirs("models/samples", exist_ok=True)
    
    # 1. Load MNIST Dataset
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"Train samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")
    
    # Save one clean sample image for each digit (0 to 9) for Home page and Model showcase
    digit_samples = {}
    for digit in range(10):
        idx = np.where(y_test == digit)[0][0]
        sample_img = x_test[idx]
        img_path = f"models/samples/digit_{digit}.png"
        Image.fromarray(sample_img).save(img_path)
        digit_samples[str(digit)] = {
            "path": img_path,
            "index": int(idx),
            "label": int(digit)
        }
    
    # Preprocess data
    x_train_norm = x_train.astype("float32") / 255.0
    x_test_norm = x_test.astype("float32") / 255.0
    
    x_train_norm = np.expand_dims(x_train_norm, -1)
    x_test_norm = np.expand_dims(x_test_norm, -1)
    
    # 2. Build model
    model = build_model()
    model.summary()
    
    # 3. Train model
    epochs = 6
    batch_size = 128
    print(f"\nTraining CNN for {epochs} epochs...")
    
    history = model.fit(
        x_train_norm,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )
    
    # 4. Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test_norm, y_test, verbose=1)
    print(f"Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")
    
    # Compute confusion matrix
    y_pred_probs = model.predict(x_test_norm)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Confusion matrix 10x10
    confusion_mtx = tf.math.confusion_matrix(y_test, y_pred).numpy().tolist()
    
    # Per-class accuracy
    per_class_acc = {}
    for d in range(10):
        total_d = np.sum(y_test == d)
        correct_d = np.sum((y_test == d) & (y_pred == d))
        per_class_acc[str(d)] = {
            "total": int(total_d),
            "correct": int(correct_d),
            "accuracy": round(float(correct_d / total_d) * 100, 2)
        }
    
    # Layer details
    layer_info = []
    for layer in model.layers:
        layer_info.append({
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": [str(s) for s in layer.output.shape],
            "params": int(layer.count_params())
        })
    
    # 5. Save model
    model_save_path = "models/mnist_cnn.keras"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # 6. Save metadata
    metadata = {
        "model_name": "Deep Convolutional Neural Network (CNN)",
        "epochs": epochs,
        "batch_size": batch_size,
        "test_accuracy": round(float(test_acc) * 100, 2),
        "test_loss": round(float(test_loss), 4),
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum([keras.backend.count_params(w) for w in model.trainable_weights])),
        "history": {
            "accuracy": [round(float(v), 4) for v in history.history["accuracy"]],
            "val_accuracy": [round(float(v), 4) for v in history.history["val_accuracy"]],
            "loss": [round(float(v), 4) for v in history.history["loss"]],
            "val_loss": [round(float(v), 4) for v in history.history["val_loss"]],
        },
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion_mtx,
        "layers": layer_info,
        "samples": digit_samples,
        "dataset_info": {
            "name": "MNIST (Modified National Institute of Standards and Technology)",
            "authors": "Yann LeCun, Corinna Cortes, Christopher J.C. Burges",
            "training_samples": int(x_train.shape[0]),
            "test_samples": int(x_test.shape[0]),
            "image_dimensions": "28 x 28 pixels (Grayscale)",
            "num_classes": 10,
            "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        }
    }
    
    with open("models/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Training metadata saved to models/training_metadata.json")

if __name__ == "__main__":
    main()
