import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 🔐 SAFE IMAGE MODEL LOADING
try:
    image_model = load_model("models/image_model.h5")
except Exception as e:
    image_model = None
    print("⚠️ Image model not found:", e)

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(img):
    if image_model is None:
        return None, "Image model not loaded"

    processed = preprocess_image(img)
    pred = image_model.predict(processed, verbose=0)
    score = float(pred[0][0])
    return score, None