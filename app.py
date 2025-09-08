import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==============================
# Load trained model & scaler
# ==============================
model = load_model("student_model.h5")
scaler = joblib.load("scaler.pkl")    # must be saved during training
encoders = joblib.load("encoders.pkl")  # optional: dictionary of LabelEncoders if you saved them

# ==============================
# App UI
# ==============================
st.title("🎓 Student Performance Classifier")
st.write("Predict student performance (High / Middle / Low) based on demographics, academics, and behavior.")

# ---------- Demographic Features ----------
gender = st.selectbox("👤 Gender", ["Male", "Female"])
nationality = st.selectbox("🌍 Nationality", 
    ["Kuwait","Lebanon","Egypt","SaudiArabia","USA","Jordan","Venezuela",
     "Iran","Tunis","Morocco","Syria","Palestine","Iraq","Lybia"])
birth_place = st.selectbox("🏠 Place of Birth", 
    ["Kuwait","Lebanon","Egypt","SaudiArabia","USA","Jordan","Venezuela",
     "Iran","Tunis","Morocco","Syria","Palestine","Iraq","Lybia"])

# ---------- Academic Features ----------
stage = st.selectbox("📚 Stage ID", ["lowerlevel","MiddleSchool","HighSchool"])
grade = st.selectbox("🎓 Grade ID", 
    ["G-01","G-02","G-03","G-04","G-05","G-06","G-07","G-08","G-09","G-10","G-11","G-12"])
section = st.selectbox("🏫 Section ID", ["A","B","C"])
topic = st.selectbox("📖 Course Topic", 
    ["English","Spanish","French","Arabic","IT","Math","Chemistry","Biology",
     "Science","History","Quran","Geology"])
semester = st.selectbox("📅 Semester", ["First","Second"])
parent = st.selectbox("👪 Parent Responsible", ["mom","father"])

# ---------- Behavioral Features ----------
raised_hand = st.slider("✋ Raised Hand (0–100)", 0, 100, 10)
resources = st.slider("📚 Visited Resources (0–100)", 0, 100, 20)
announcements = st.slider("📢 Viewing Announcements (0–100)", 0, 100, 5)
discussion = st.slider("💬 Discussion Groups (0–100)", 0, 100, 5)
parent_survey = st.radio("📝 Parent Answering Survey", ["Yes","No"])
school_satisfaction = st.radio("🏫 Parent School Satisfaction", ["Yes","No"])
absence = st.radio("🗓 Absence Days", ["Under-7","Above-7"])

# ==============================
# Preprocessing
# ==============================
# Collect features in same order used for training
features = [gender, nationality, birth_place, stage, grade, section, topic, semester, parent,
            raised_hand, resources, announcements, discussion, parent_survey, school_satisfaction, absence]

# Encode categorical variables (must match training encoding)
encoded_features = []
for i, col in enumerate(features):
    if isinstance(col, str):  # categorical
        # use saved LabelEncoder
        le = encoders[i]
        encoded_val = le.transform([col])[0]
        encoded_features.append(encoded_val)
    else:
        encoded_features.append(col)

X = np.array([encoded_features])

# Scale numeric features
X_scaled = scaler.transform(X)

# ==============================
# Prediction
# ==============================
if st.button("🔍 Predict Performance"):
    pred = model.predict(X_scaled)
    cls = np.argmax(pred)
    classes = ["Low","Middle","High"]
    st.success(f"✅ Predicted Student Performance: **{classes[cls]}**")
