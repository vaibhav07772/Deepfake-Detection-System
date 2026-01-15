# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # 🔐 SAFE VIDEO MODEL LOADING
# try:
#     video_model = load_model("models/video_model.h5")
# except Exception as e:
#     video_model = None
#     print("⚠️ Video model not found:", e)

# def extract_frames(video_path, max_frames=20):
#     cap = cv2.VideoCapture(video_path)
#     frames = []

#     while len(frames) < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.resize(frame, (224, 224))
#         frame = frame / 255.0
#         frames.append(frame)

#     cap.release()
#     return np.array(frames)

# def predict_video(video_path):
#     if video_model is None:
#         return None, "Video model not loaded"

#     frames = extract_frames(video_path)

#     if len(frames) == 0:
#         return None, "No frames extracted from video"

#     preds = video_model.predict(frames, verbose=0)
#     score = float(preds.mean())
#     return score, None

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 🔐 SAFE VIDEO MODEL LOADING
try:
    video_model = load_model("models/video_model.h5")
except Exception as e:
    video_model = None
    print("⚠️ Video model not found:", e)


def extract_frames(video_path, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()
    return np.array(frames)


def predict_video(video_path):
    if video_model is None:
        return None, "Video model not loaded"

    frames = extract_frames(video_path)

    if len(frames) == 0:
        return None, "No frames extracted from video"

    # 🔥 IMPORTANT FIX (batch dimension add)
    frames = np.expand_dims(frames, axis=0)
    # shape now → (1, frames, 224, 224, 3)

    preds = video_model.predict(frames, verbose=0)
    score = float(preds[0][0])

    return score, None