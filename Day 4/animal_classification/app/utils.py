# app/utils.py
from PIL import Image
import numpy as np
from pathlib import Path
import joblib

IMG_SIZE = (64, 64)  # must match training

MODEL_DIR = Path(__file__).parent / "models"

def preprocess_image_bytes(file_bytes, img_size=IMG_SIZE):
    """
    Input: raw bytes from uploaded file
    Returns: 1D numpy array suitable for model input (scaled as training did)
    """
    img = Image.open(file_bytes).convert("RGB")
    img = img.resize(img_size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr.flatten().reshape(1, -1)

def load_model_by_name(name):
    """
    name: 'logistic', 'knn', 'gaussian_nb'
    """
    name_map = {
        "logistic": "logistic.pkl",
        "knn": "knn.pkl",
        "gaussian": "gaussian_nb.pkl",
        "gaussian_nb": "gaussian_nb.pkl"
    }
    model_file = name_map.get(name)
    if model_file is None:
        raise ValueError("Unknown model name")
    model_path = MODEL_DIR / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Train models first.")
    model = joblib.load(model_path)
    return model

def load_auxiliary():
    """Load scaler and label map"""
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    label_map = joblib.load(MODEL_DIR / "label_map.joblib")
    return scaler, label_map
