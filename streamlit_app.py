import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from PIL import Image

# Load logo
logo = Image.open("image.png")
st.image(logo, width=300)

# Load and train models once
@st.cache_resource
def train_models():
    df = pd.read_csv("gym_floor_raw.csv")
    df.columns = [col.strip() for col in df.columns]

    required_columns = {
        "Quoted_Price", "Size_sqft", "Coats", "Labor_Hours", "Distance", "Concurrent_Job", "GP_Percent"
    }
    if not required_columns.issubset(set(df.columns)):
        st.error(f"Missing one or more required columns: {required_columns}")
        return None, None

    df = df.dropna(subset=list(required_columns))

    # Train labor hours prediction model
    X_hours = df[["Size_sqft", "Coats", "Distance", "Concurrent_Job"]]
    y_hours = df["Labor_Hours"]
    labor_model = RandomForestRegressor()
    labor_model.fit(X_hours, y_hours)

    # Train GP% prediction model
    df["Quoted_Price_per_sqft"] = df["Quoted_Price"] / df["Size_sqft"]
    X_gp = df[["Quoted_Price_per_sqft", "Size_sqft", "Coats", "Labor_Hours", "Distance", "Concurrent_Job"]]
    y_gp = df["GP_Percent"]
    gp_model = RandomForestRegressor()
    gp_model.fit(X_gp, y_gp)

    return labor_model, gp_model

labor_model, gp_model = train_models()

# App UI
st.title("Gym Floor Pricing Estimator")

square_feet = st.number_input("Enter square footage", min_value=4000, max_value=100000, step=100)
coats = st.selectbox("Number of coats", options=[1, 2])
distance = st.number_input("Distance to job site (miles)", min_value=1, max_value=500)
concurrent_job = st.selectbox("Is this a concurrent job?", options=["No", "Yes"])
concurrent_job_flag = 1 if concurrent_job == "Yes" else 0

if st.button("Estimate Price"):
    # Step 1: Predict labor hours
    labor_input = [[square_feet, coats, distance, concurrent_job_flag]]
    predicted_labor_hours = labor_model.predict(labor_input)[0]

    # Step 2: Price loop to ensure GP% >= 45% and respect floor pricing
    floor_price = 0.67 if coats == 2 else 0.48
    best_price_per_sqft = floor_price
    
    for price_per_sqft in np.arange(floor_price, 2.0, 0.01):
        total_price = price_per_sqft * square_feet
        gp_input = [[
            price_per_sqft,
            square_feet,
            coats,
            predicted_labor_hours,
            distance,
            concurrent_job_flag
        ]]
        predicted_gp = gp_model.predict(gp_input)[0]
        if predicted_gp >= 45.0:
            best_price_per_sqft = price_per_sqft
            break

    final_total = best_price_per_sqft * square_feet

    st.subheader("Estimate")
    st.write(f"**Predicted labor hours:** {predicted_labor_hours:.1f}")
    st.write(f"**Price per sq ft:** ${best_price_per_sqft:.2f}")
    st.write(f"**Total estimated price:** ${final_total:,.2f}")
