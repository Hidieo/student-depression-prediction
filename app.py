import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model
model = joblib.load("mental_health_model.pkl")

st.set_page_config(page_title="Student Depression Predictor", layout="centered")
st.title("🎓 Mental Health Prediction for Students")

st.markdown("**Prediksi risiko depresi berdasarkan perilaku dan kebiasaan mahasiswa.**")

# Form input
gender = st.selectbox("Gender", ["Male", "Female"])
academic_pressure = st.slider("Academic Pressure", 0, 5, 3)
study_hours = st.slider("Study Hours per Week", 0, 12, 5)

suicidal_thought = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
degree = st.selectbox("Degree", ["'Class 12'", "B.Ed", "B.Com", "B.Arch", "BCA", "MSc",
                                 "B.Tech", "MCA", "M.Tech", "BHM", "BSc", "M.Ed", "B.Pharm", 
                                 "M.Com", "MBBS", "BBA", "LLB", "BE", "BA", "M.Pharm", "MD",
                                 "MBA", "MA", "PhD", "LLM", "MHM", "ME", "Others"])

sleep_duration = st.selectbox("Sleep Duration", [
    "'Less than 5 hours'", "'5-6 hours'", "'7-8 hours'", "'More than 8 hours'", "Others"
])

dietary = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy", "Others"])

financial_stress = st.selectbox("Financial Stress", ['1.0', '2.0', '3.0', '4.0', '5.0', '?'])

# Buat DataFrame dari input
input_df = pd.DataFrame([{
    'Gender': gender,
    'Academic Pressure': academic_pressure,
    'Work/Study Hours': study_hours,
    'Have you ever had suicidal thoughts ?': suicidal_thought,
    'Family History of Mental Illness': family_history,
    'Degree': degree,
    'Sleep Duration': sleep_duration,
    'Dietary Habits': dietary,
    'Financial Stress': financial_stress
}])

if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    result = "🟠 Depression" if prediction == 1 else "🟢 Not Depression"
    st.subheader("Result:")
    if prediction == 0:
      st.success(result)
    else:
      st.error(result)
