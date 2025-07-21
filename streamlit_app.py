import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load and prepare data
@st.cache_resource
def train_model():
    df = pd.read_csv("gym_floor_raw.csv")
    st.write("Column names in CSV:", df.columns.tolist())

    df.columns = [col.strip() for col in df.columns]  # Clean column names

    required_columns = {"Quoted_Price", "Size_sqft", "Coats", "Labor_Hours", "Distance", "Concurrent_Job"}
    if not required_columns.issubset(df.columns):
        st.error(f"Missing one or more required columns: {required_columns}")
        return None

    df = df.dropna(subset=["Quoted_Price", "Size_sqft", "Coats", "Labor_Hours", "Distance", "Concurrent_Job"])

    X = df[["Size_sqft", "Coats", "Labor_Hours", "Distance", "Concurrent_Job"]]
    y = df["Quoted_Price"]

    model = RandomForestRegressor()
    model.fit(X, y)

    return model

# Train once and reuse
model = train_model()

# App UI
st.title("Gym Floor Pricing Estimator")

square_feet = st.number_input("Enter square footage", min_value=100, max_value=20000, step=100)
coats = st.number_input("Enter number of coats", min_value=1, max_value=3, step=1)
labor_hours = st.number_input("Enter estimated labor hours", min_value=1, max_value=100, step=1)
distance = st.number_input("Enter distance to site (miles)", min_value=1, max_value=300, step=1)
concurrent_job = st.selectbox("Is this job concurrent with another?", ["No", "Yes"])
concurrent_job_flag = 1 if concurrent_job == "Yes" else 0

if st.button("Estimate Price"):
    if model:
        features = [[square_feet, coats, labor_hours, distance, concurrent_job_flag]]
        prediction = model.predict(features)
        st.success(f"Estimated price: ${prediction[0]:,.2f}")
    else:
        st.error("Model could not be trained.")
