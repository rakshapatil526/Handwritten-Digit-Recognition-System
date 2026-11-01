"""
Image preprocessing pipeline for Handwritten Digit Recognition.
Converts freehand canvas drawings and uploaded images to the exact MNIST format:
- 28x28 dimensions
- Grayscale, white digit on black background (0 = black, 255 = white)
- Aspect-ratio preserved scaling into a 20x20 bounding box
- Center of mass alignment
- Normalized float32 [0.0, 1.0] tensor
"""

import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift

def preprocess_canvas_image(image_data, invert_colors=False):
    """
    Process an RGBA numpy array from streamlit-drawable-canvas.
    
    Parameters:
    - image_data: np.ndarray of shape (H, W, 4) from canvas
    - invert_colors: bool, set to True if drawn as black digit on white canvas
    
    Returns:
    - processed_tensor: np.ndarray of shape (1, 28, 28, 1) in [0.0, 1.0]
    - preview_pil: PIL Image (28x28) of the preprocessed digit
    - is_blank: bool, True if no digit detected
    """
    if image_data is None:
        return None, None, True
    
    # Extract alpha or RGB
    if image_data.ndim == 3 and image_data.shape[2] == 4:
        # Check if alpha channel has drawing
        alpha = image_data[:, :, 3]
        rgb = image_data[:, :, :3]
        
        # If alpha is uniform 255, convert RGB to grayscale
        if np.all(alpha == 255) or np.mean(alpha) > 250:
            gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            # If average is high (white canvas with dark drawing), invert
            if np.mean(gray) > 127 or invert_colors:
                gray = 255 - gray
        else:
            # Drawn on transparent background with white/colored pen
            gray = alpha
    elif image_data.ndim == 3 and image_data.shape[2] == 3:
        gray = np.dot(image_data[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        if np.mean(gray) > 127 or invert_colors:
            gray = 255 - gray
    else:
        gray = np.array(image_data).astype(np.uint8)
        if np.mean(gray) > 127 or invert_colors:
            gray = 255 - gray

    return _fit_to_mnist_format(gray)

def preprocess_uploaded_image(uploaded_file):
    """
    Process an uploaded user image (JPG/PNG).
    """
    pil_img = Image.open(uploaded_file).convert("L")
    img_arr = np.array(pil_img)
    
    # In MNIST, background is dark (0) and digit is light (255)
    # If uploaded image has light background, invert it
    if np.mean(img_arr) > 127:
        img_arr = 255 - img_arr
        
    return _fit_to_mnist_format(img_arr)

def _fit_to_mnist_format(gray_arr):
    """
    Core function that fits any grayscale digit image into MNIST standard:
    1. Threshold noise
    2. Crop bounding box
    3. Scale into 20x20 box preserving aspect ratio
    4. Center in 28x28 canvas using Center of Mass
    5. Normalize to [0, 1]
    """
    # Threshold out noise
    thresh = 35
    gray_arr[gray_arr < thresh] = 0
    
    # Check if image is blank
    if np.max(gray_arr) < 25 or np.sum(gray_arr > 30) < 15:
        # Blank canvas
        blank_28 = np.zeros((28, 28), dtype=np.uint8)
        blank_pil = Image.fromarray(blank_28)
        blank_tensor = np.zeros((1, 28, 28, 1), dtype=np.float32)
        return blank_tensor, blank_pil, True

    # Find bounding box
    coords = np.argwhere(gray_arr > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Crop
    digit_crop = gray_arr[y_min:y_max + 1, x_min:x_max + 1]
    h, w = digit_crop.shape
    
    if h == 0 or w == 0:
        blank_28 = np.zeros((28, 28), dtype=np.uint8)
        return np.zeros((1, 28, 28, 1), dtype=np.float32), Image.fromarray(blank_28), True
    
    # Resize preserving aspect ratio inside a 20x20 box
    crop_pil = Image.fromarray(digit_crop)
    if h > w:
        new_h = 20
        new_w = max(1, int(round((w * 20.0) / h)))
    else:
        new_w = 20
        new_h = max(1, int(round((h * 20.0) / w)))
        
    resized_digit = crop_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_arr = np.array(resized_digit)
    
    # Paste into center of a 28x28 black canvas
    canvas_28 = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas_28[start_y:start_y + new_h, start_x:start_x + new_w] = resized_arr.astype(np.float32)
    
    # Center of mass alignment (MNIST standard: center of mass at (14, 14))
    cy, cx = center_of_mass(canvas_28)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_y = 14.0 - cy
        shift_x = 14.0 - cx
        
        # Limit shift so strokes don't go outside border
        shift_y = np.clip(shift_y, -3.5, 3.5)
        shift_x = np.clip(shift_x, -3.5, 3.5)
        
        canvas_28 = shift(canvas_28, [shift_y, shift_x], mode="constant", cval=0.0)
    
    # Clip and convert
    canvas_28 = np.clip(canvas_28, 0.0, 255.0)
    preview_pil = Image.fromarray(canvas_28.astype(np.uint8))
    
    # Normalize to [0.0, 1.0]
    normalized_tensor = (canvas_28 / 255.0).astype(np.float32)
    normalized_tensor = np.expand_dims(normalized_tensor, axis=(0, -1)) # Shape: (1, 28, 28, 1)
    
    return normalized_tensor, preview_pil, False
