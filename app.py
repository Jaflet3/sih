# ✅ Install dependencies before running
# pip install streamlit opencv-python-headless deepface pandas

import os
import zipfile
import cv2
import pandas as pd
from deepface import DeepFace
from datetime import datetime
import streamlit as st

st.title("🎓 Smart Attendance System (SIH v2)")

# --------------------------
# STEP 1: Upload Student ZIP
# --------------------------
uploaded_zip = st.file_uploader("📂 Upload Student Images ZIP", type=["zip"])

students_db = {}
extract_folder = "students"

if uploaded_zip is not None:
    # Save uploaded file
    zip_path = os.path.join("students.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.read())
    
    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    
    st.success("✅ Extracted student images!")
    student_files = os.listdir(extract_folder)
    st.write("Enrolled Students:", student_files)

    # Build DB
    for file in student_files:
        if file.lower().endswith((".jpg", ".png")):
            name = os.path.splitext(file)[0]
            students_db[name] = os.path.join(extract_folder, file)

# --------------------------
# STEP 2: Attendance Function
# --------------------------
def mark_attendance(class_image):
    attendance = []

    # Detect faces
    try:
        faces = DeepFace.extract_faces(img_path=class_image, enforce_detection=False)
    except Exception as e:
        st.error(f"⚠ Error detecting faces: {e}")
        faces = []
    
    matched_students = set()

    for face_obj in faces:
        face_img = face_obj['face']
        for name, path in students_db.items():
            if name in matched_students:
                continue
            try:
                result = DeepFace.verify(img1_path=face_img, img2_path=path,
                                         enforce_detection=False, model_name="Facenet")
                if result['verified']:
                    matched_students.add(name)
                    attendance.append({"Name": name, "Status": "Present", "Time": datetime.now()})
            except Exception as
