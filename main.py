import streamlit as st
import cv2
import numpy as np
import tempfile

from utils.image_utils import predict_image
from utils.video_utils import predict_video

st.set_page_config(page_title="Deepfake Detector", layout="wide")

st.title("🕵️ Deepfake Detector")
st.write("Detect deepfake images and videos using advanced AI models")

col1, col2 = st.columns(2)

# ================= IMAGE =================
with col1:
    st.subheader("Deepfake Image Detector")
    img_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, caption="Uploaded Image", width='stretch')

        score, error = predict_image(img)

        if error:
            st.warning(f"⚠️ {error}")
        elif score > 0.7:
            st.error(f"❌ FAKE IMAGE\nConfidence: {score:.2f}")
        else:
            st.success(f"✅ REAL IMAGE\nConfidence: {1-score:.2f}")

# ================= VIDEO =================
with col2:
    st.subheader("Deepfake Video Detector")
    vid_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

    if vid_file:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(vid_file.read())
        temp.close()

        score, error = predict_video(temp.name)

        if error:
            st.warning(f"⚠️ {error}")
        elif score > 0.6:
            st.error(f"❌ FAKE VIDEO\nConfidence: {score:.2f}")
        else:
            st.success(f"✅ REAL VIDEO\nConfidence: {1-score:.2f}")

st.caption("👨‍💻 Built by Vaibhav Singh | Deepfake Detection Project")