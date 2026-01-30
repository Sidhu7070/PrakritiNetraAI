# streamlit_final_app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
import numpy as np
import cv2
import json
import os
import tempfile
import pandas as pd


# PATH CONFIGURATION

MODEL_DIR = "models2"
MODEL_PATH = os.path.join(MODEL_DIR, "plastic_cnn.pth")
META_PATH = os.path.join(MODEL_DIR, "meta_classes.json")

st.set_page_config(page_title="🌿 PrakritiNetraAI", layout="wide")


# BACKGROUND & THEME

st.markdown(f"""
<style>
.stApp {{
    background-image: url("file:///C:/Users/sidha/OneDrive/Desktop/Main ML/back3.jpg");  /* path to your image */
    background-size: cover;       /* stretch image to cover whole screen */
    background-position: center;  /* center the image */
    background-repeat: no-repeat; /* don’t repeat */
    background-attachment: fixed; /* stays fixed when scrolling */
}}
h1, h2, h3, h4 {{
    color: #1b5e20;  /* adjust text color for visibility */
    text-shadow: 1px 1px 2px white;  /* optional: makes text readable */
}}
</style>
""", unsafe_allow_html=True)



# MODEL SETUP

if not os.path.exists(META_PATH):
    meta = {"0": "non-plastic", "1": "plastic"}
    with open(META_PATH, "w") as f:
        json.dump(meta, f)

with open(META_PATH, "r") as f:
    meta = json.load(f)

classes = [meta[str(i)] for i in range(len(meta))]

weights = ResNet34_Weights.DEFAULT
model = resnet34(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# APP UI

st.title("🌿 PrakritiNetraAI – Plastic vs Non-Plastic Detection")
st.write("Detect plastic waste from images using **CNN + ResNet34 Hybrid Model**")

option = st.radio("Select mode:",
                  ["📸 Single Image Detection", "🎥 Real-Time Detection", "🧩 Multi-Object Detection"])


# 1️⃣ SINGLE IMAGE DETECTION

if option == "📸 Single Image Detection":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_t)
            probs = F.softmax(out, dim=1)[0].cpu().numpy()

        pred_idx = np.argmax(probs)
        label = classes[pred_idx]
        confidence = probs[pred_idx]

        color = "#ff0000" if label == "plastic" else "#00cc44"
        st.markdown(f"""
        <div style="padding:15px; border-radius:10px; background-color:black; color:white; text-align:center; border:3px solid {color}">
        <h2>Prediction: {label.upper()} </h2>
        <h4>Confidence: {confidence*100:.2f}%</h4>
        </div>
        """, unsafe_allow_html=True)


# 2️⃣ REAL-TIME DETECTION

elif option == "🎥 Real-Time Detection":
    st.write("Use your webcam to detect a single object in real-time.")
    cam = st.camera_input("Capture Image")

    if cam:
        img = Image.open(cam).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_t)
            probs = F.softmax(out, dim=1)[0].cpu().numpy()

        pred_idx = np.argmax(probs)
        label = classes[pred_idx]
        confidence = probs[pred_idx]

        color = "#ff0000" if label == "plastic" else "#00cc44"
        st.markdown(f"""
        <div style="padding:15px; border-radius:10px; background-color:black; color:white; text-align:center; border:3px solid {color}">
        <h2>Prediction: {label.upper()}</h2>
        <h4>Confidence: {confidence*100:.2f}%</h4>
        </div>
        """, unsafe_allow_html=True)


# 3️⃣ MULTI-OBJECT DETECTION (with Live Option)

elif option == "🧩 Multi-Object Detection":
    input_choice = st.radio("Select input:", ["📂 Upload Image", "📷 Capture from Camera"])

    if input_choice == "📂 Upload Image":
        file = st.file_uploader("Upload image with multiple objects", type=["jpg", "jpeg", "png"])
        if file:
            img = Image.open(file).convert("RGB")
    else:
        cam_img = st.camera_input("Capture Image")
        if cam_img:
            img = Image.open(cam_img).convert("RGB")

    if 'img' in locals():
        np_img = np.array(img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 40 or h < 40:
                continue
            roi = np_img[y:y+h, x:x+w]
            roi_img = Image.fromarray(roi)
            inp = transform(roi_img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp)
                probs = F.softmax(out, dim=1)[0].cpu().numpy()
            pred_idx = np.argmax(probs)
            label = classes[pred_idx]
            conf = probs[pred_idx]

            color = (0, 255, 0) if label == "non-plastic" else (255, 0, 0)
            cv2.rectangle(np_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(np_img, f"{label} ({conf*100:.1f}%)", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            detected.append({"Label": label, "Confidence": f"{conf*100:.2f}%"})

        st.image(np_img, caption="Detected Objects", use_container_width=True)
        if len(detected) > 0:
            st.dataframe(pd.DataFrame(detected))
        else:
            st.warning("No detectable objects found.")
