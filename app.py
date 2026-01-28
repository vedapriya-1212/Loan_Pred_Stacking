import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load PIPELINE / MODEL

model = joblib.load("stacking_pipeline.pkl")   # or stacking_model.pkl if you kept that name

# Page Config

st.set_page_config(page_title="Smart Loan Approval System", layout="wide")

# Title & Description

st.title("🎯 Smart Loan Approval System – Stacking Model")

st.write("""
This system uses a **Stacking Ensemble Machine Learning model** to predict whether a loan will be approved  
by combining multiple ML models for better decision making.
""")

# Sidebar Inputs

st.sidebar.header("📋 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapp_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0)

credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

# Manual Encoding (Same As Before)

credit_history = 1 if credit_history == "Yes" else 0
employment = 0 if employment == "Salaried" else 1

property_map = {
    "Urban": 2,
    "Semi-Urban": 1,
    "Rural": 0
}

property_area = property_map[property_area]

# Create Base Input (7 features)

input_array = np.array([[

    app_income,
    coapp_income,
    loan_amount,
    loan_term,
    credit_history,
    employment,
    property_area

]])

# FIX FEATURE MISMATCH HERE

# Find how many features model expects
expected_features = model.named_steps['imputer'].statistics_.shape[0]

current_features = input_array.shape[1]

if current_features < expected_features:

    diff = expected_features - current_features

    padding = np.zeros((1, diff))

    input_array = np.hstack((input_array, padding))

# Model Architecture Display

st.subheader("🧩 Stacking Model Architecture")

st.info("""
### Base Models Used:
• Logistic Regression  
• Decision Tree  
• Random Forest  

### Meta Model Used:
• Logistic Regression  
""")

# Prediction Button

if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    # Final Prediction

    final_pred = model.predict(input_array)[0]
    confidence = model.predict_proba(input_array)[0][1] * 100

    # Output Section

    st.subheader("📊 Prediction Results")

    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"📈 Confidence Score: **{confidence:.2f}%**")

    # Business Explanation

    st.subheader("💼 Business Explanation")

    if final_pred == 1:

        st.success("""
Based on income level, credit history, and combined predictions from multiple machine learning models,  
the applicant shows strong repayment potential.

Therefore, the stacking model predicts **loan approval**.
""")

    else:

        st.error("""
Based on financial risk factors, credit history, and combined model analysis,  
the applicant shows higher default risk.

Therefore, the stacking model predicts **loan rejection**.
""")
