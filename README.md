# Deepfake Image & Video Detector

An end-to-end Deepfake Detection web application built using **Deep Learning, OpenCV, and Streamlit**.  
This project detects whether an uploaded **image or video is REAL or FAKE (deepfake)** and shows a confidence score.

## 🚀 Features

- ✅ Detect deepfake **images**
- ✅ Detect deepfake **videos**
- ✅ Separate pipelines for image and video detection
- ✅ CNN-based image model
- ✅ CNN + LSTM-based video model
- ✅ Interactive Streamlit UI
- ✅ Confidence score visualization
- ✅ Error handling for missing models or corrupted files

---

## 🧠 How This Project Works

### 🔹 Image Detection Pipeline
1. User uploads an image
2. Image is resized to **224×224**
3. Pixel values are normalized (0–1)
4. Image is passed to a **CNN model**
5. Model outputs a probability score:
   - `> 0.7` → ❌ Fake Image
   - `≤ 0.7` → ✅ Real Image

---

### 🔹 Video Detection Pipeline
1. User uploads a video
2. Video frames are extracted using OpenCV
3. Frames are resized and normalized
4. Frames are passed as a sequence to a **CNN + LSTM model**
5. Model predicts whether the video is:
   - ❌ Fake Video
   - ✅ Real Video

---

## 🏗️ Project Structure
deepfake-detector/
│
├── main.py # Streamlit application
├── models/
│ ├── image_model.h5 # CNN image model (baseline)
│ └── video_model.h5 # CNN + LSTM video model (baseline)
│
├── utils/
│ ├── image_utils.py # Image preprocessing & prediction
│ └── video_utils.py # Video frame extraction & prediction
│
├── requirements.txt
└── README.md

## How to Run
```bash
pip install -r requirements.txt

streamlit run app.py


## 🧪 Models Used

### Image Model (Baseline CNN)
- Conv2D
- Flatten
- Dense (Sigmoid output)

### Video Model (Baseline CNN + LSTM)
- TimeDistributed Conv2D
- TimeDistributed Flatten
- LSTM
- Dense (Sigmoid output)

⚠️ **Note:**  
These are **baseline (dummy) models** created for demonstrating the full deepfake detection pipeline.  
They are **not trained on real deepfake datasets**, so prediction accuracy is limited.

---

## ⚠️ Current Limitations

- Models are **not trained** on real datasets like FaceForensics++
- Predictions may be inaccurate
- Confidence scores may be close to 0.5
- Intended for **learning, demo, and system design purposes**

---

## 🔮 Future Improvements

- 🔥 Train models on **FaceForensics++ dataset**
- 🔥 Use pretrained architectures (Xception, EfficientNet)
- 🔥 Improve video detection using frame-level voting
- 🔥 Add face detection before classification
- 🔥 Deploy on cloud (Render / HuggingFace Spaces)
- 🔥 Add performance metrics (accuracy, precision, recall)

---

## 💡 Use Cases

- Academic projects
- Learning Deepfake Detection pipelines
- Demonstration for interviews
- Understanding CNN + LSTM workflows
- Foundation for production-grade deepfake systems

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit

---

## 👨‍💻 Author

**Vaibhav Singh**  
Deep Learning | NLP | Computer Vision  
📌 Built as a learning-focused Deepfake Detection project

---

## 📌 Disclaimer

This project uses **baseline models** for demonstration purposes only.  
It is **not intended for real-world forensic or legal usage** without proper training and evaluation.
