import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction System")

# ------------------------
# Load Model & Columns
# ------------------------
MODEL_PATH = "model/logistic_model.pkl"
COLUMNS_PATH = "model/columns.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
    st.error("Model not found! Please train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)
columns = joblib.load(COLUMNS_PATH)

st.markdown("---")
st.sidebar.header("🔧 Customer Information")

# ⚠ These must match actual numeric column names in your dataset
tenure = st.sidebar.number_input("Tenure", 0, 100, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 500.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# ------------------------
# Prediction
# ------------------------
if st.button("🔍 Predict Churn"):

    # Create empty dataframe with same columns as training
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0

    # Fill numeric values if present
    if 'tenure' in input_df.columns:
        input_df['tenure'] = tenure

    if 'MonthlyCharges' in input_df.columns:
        input_df['MonthlyCharges'] = monthly_charges

    if 'TotalCharges' in input_df.columns:
        input_df['TotalCharges'] = total_charges

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("### 📈 Prediction Result")

    if prediction == 1:
        st.error("⚠ Customer is likely to churn")
    else:
        st.success("✅ Customer is unlikely to churn")

    st.write(f"Churn Probability: {probability:.2%}")
    st.progress(int(probability * 100))

st.markdown("---")
st.caption("Built using Logistic Regression & Streamlit")