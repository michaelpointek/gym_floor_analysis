import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model artifacts
model = joblib.load("model_rf.pkl")
label_enc = joblib.load("label_encoder.pkl")
X_columns = joblib.load("X_columns.pkl")

st.title("Gym Floor Pricing Simulator")
st.markdown("Aim: Recommend optimal $/sqft to achieve 45% GP or higher")

# Inputs
sqft = st.number_input("Size (sqft)", min_value=1000, value=6000, step=100)
coats = st.selectbox("Number of Coats", [1, 2])
distance = st.number_input("Distance to Site (miles)", min_value=0, value=25)
labor = st.number_input("Estimated Labor Hours", min_value=1.0, value=16.0)
am = st.selectbox("Account Manager", sorted(label_enc.classes_))
concurrent = st.selectbox("Is this a concurrent job?", ["Yes", "No"])

# Encode inputs
am_encoded = label_enc.transform([am])[0]
concurrent_flag = 1 if concurrent == "Yes" else 0

input_data = {
    "Size_sqft": sqft,
    "Coats": coats,
    "Distance": distance,
    "Labor_Hours": labor,
    "AM": am_encoded,
    "Concurrent_Job": concurrent_flag
}

# Prediction function
def get_best_price_per_sqft(job_input, model, X_columns, target_gp=0.45):
    base_features = pd.DataFrame([job_input])
    sqft = job_input['Size_sqft']
    price_range = np.arange(0.40, 1.20, 0.01)

    for p in price_range:
        total_price = round(p * sqft, 2)
        row = base_features.copy()
        row['Quoted_Price'] = total_price
        row = row.reindex(columns=X_columns, fill_value=0)
        gp = model.predict(row)[0]
        if gp >= target_gp:
            return p, total_price, gp

    return None, None, None

# Show result
if st.button("Recommend Price"):
    price_per_sqft, total, gp = get_best_price_per_sqft(input_data, model, X_columns)
    if price_per_sqft:
        st.success(f"✔️ Quote: ${total} (${price_per_sqft}/sqft)")
        st.write(f"Predicted GP%: {gp:.2%}")
    else:
        st.error("❌ No price in range meets 45% GP target.")
