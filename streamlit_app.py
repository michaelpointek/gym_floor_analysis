import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from PIL import Image

logo = Image.open("image.png")  # Replace with your logo filename
st.image(logo, width=300)  # Adjust width as needed

# Load and train model once
@st.cache_resource
def train_model():
    df = pd.read_csv("gym_floor_raw.csv")
    df.columns = [col.strip() for col in df.columns]

    required_columns = {"Quoted_Price", "Size_sqft", "Coats", "Labor_Hours", "Distance", "Concurrent_Job"}
    if not required_columns.issubset(set(df.columns)):
        st.error(f"Missing one or more required columns: {required_columns}")
        return None

    df = df.dropna(subset=list(required_columns))

    X = df[["Size_sqft", "Coats", "Labor_Hours", "Distance", "Concurrent_Job"]]
    y = df["Quoted_Price"]

    model = RandomForestRegressor()
    model.fit(X, y)

    return model

model = train_model()

# App UI
st.title("Gym Floor Pricing Estimator")

square_feet = st.number_input("Enter square footage", min_value=4000, max_value=100000, step=100)
coats = st.selectbox("Number of coats", options=[1, 2])
labor_hours = st.number_input("Estimated labor hours", min_value=0, max_value=500)
distance = st.number_input("Distance to job site (miles)", min_value=1, max_value=500)
concurrent_job = st.selectbox("Is this a concurrent job?", options=["No", "Yes"])
concurrent_job_flag = 1 if concurrent_job == "Yes" else 0

if st.button("Estimate Price"):
    input_features = [[square_feet, coats, labor_hours, distance, concurrent_job_flag]]
    predicted_total = model.predict(input_features)[0]

    price_per_sqft = predicted_total / square_feet

    # Enforce minimum pricing
    min_rate = 0.67 if coats == 2 else 0.48
    price_per_sqft = max(price_per_sqft, min_rate)
    final_total = price_per_sqft * square_feet

    st.subheader("Estimate")
    st.write(f"**Price per sq ft:** ${price_per_sqft:.2f}")
    st.write(f"**Total estimated price:** ${final_total:,.2f}")
