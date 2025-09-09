# ✅ Install dependencies before running:
# pip install streamlit deepface pandas pillow

import os
import zipfile
import pandas as pd
from deepface import DeepFace
from datetime import datetime
from PIL import Image
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
    zip_path = "students.zip"
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.read())
    
    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    
    st.success("✅ Extracted student images!")
    student_files = os.listdir(extract_folder)
    st.write("📌 Enrolled Students:", student_files)

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

    # Detect faces from classroom image
    try:
        faces = DeepFace.extract_faces(img_path=class_image, enforce_detection=False)
    except Exception as e:
        st.error(f"⚠ Error detecting faces: {e}")
        return []

    matched_students = set()

    for face_obj in faces:
        face_img = face_obj['face']  # numpy array of the face
        for name, path in students_db.items():
            if name in matched_students:
                continue
            try:
                result = DeepFace.verify(
                    img1_path=face_img,
                    img2_path=path,
                    enforce_detection=False,
                    model_name="Facenet"
                )
                if result['verified']:
                    matched_students.add(name)
                    attendance.append({
                        "Name": name,
                        "Status": "Present",
                        "Time": datetime.now()
                    })
            except Exception as e:
                st.warning(f"⚠ Error verifying {name}: {e}")

    # Mark absent students
    for name in students_db:
        if name not in matched_students:
            attendance.append({
                "Name": name,
                "Status": "Absent",
                "Time": datetime.now()
            })
    
    return attendance

# --------------------------
# STEP 3: Upload Classroom Image
# --------------------------
uploaded_class = st.file_uploader("🖼️ Upload Classroom Image", type=["jpg", "png"])

if uploaded_class is not None and students_db:
    class_path = "class.jpg"
    with open(class_path, "wb") as f:
        f.write(uploaded_class.read())

    st.image(Image.open(class_path), caption="Classroom Image", use_column_width=True)

    if st.button("📌 Mark Attendance"):
        records = mark_attendance(class_path)
        if records:
            df = pd.DataFrame(records)
            st.dataframe(df)

            # Save CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Attendance CSV", csv, "attendance.csv", "text/csv")
        else:
            st.error("⚠ No attendance data recorded.")
