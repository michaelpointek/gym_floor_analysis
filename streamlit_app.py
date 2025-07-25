import os
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------------------------
# 1. LOGGING FUNCTION
# -----------------------------------------------------------------------------
def log_prediction(
    square_feet: float,
    coats: int,
    distance: float,
    concurrent_job: int,
    predicted_labor: float,
    price_per_sqft: float,
    total_price: float,
    logfile: str = "labor_predictions_log.csv"
):
    """Append this run’s inputs + outputs to the CSV log."""
    # Ensure file exists and has a header row
    file_exists = os.path.isfile(logfile)
    with open(logfile, "a", newline="") as f:
        writer = pd.io.common.csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "Size_sqft",
                "Coats",
                "Distance",
                "Concurrent_Job",
                "Predicted_Labor_Hours",
                "Price_per_sqft",
                "Total_Price"
            ])
        writer.writerow([
            datetime.now().isoformat(),
            square_feet,
            coats,
            distance,
            concurrent_job,
            round(predicted_labor, 2),
            round(price_per_sqft, 2),
            round(total_price, 2)
        ])


# -----------------------------------------------------------------------------
# 2. MODEL TRAINING (cached)
# -----------------------------------------------------------------------------
@st.cache_resource(ttl=3600)
def train_models():
    # -- 2.1 Train the labor-hours model on your log CSV
    log_df = pd.read_csv("labor_predictions_log.csv")
    log_df = log_df.dropna(subset=[
        "Size_sqft", "Coats", "Distance", "Concurrent_Job", "Predicted_Labor_Hours"
    ])

    X_labor = log_df[["Size_sqft", "Coats", "Distance", "Concurrent_Job"]]
    y_labor = log_df["Predicted_Labor_Hours"]
    labor_model = LinearRegression().fit(X_labor, y_labor)

    # -- 2.2 Train the GP% model on your raw gym_floor data
    raw = pd.read_csv("gym_floor_raw.csv")
    raw.columns = [c.strip() for c in raw.columns]
    required = {
        "Quoted_Price", "Size_sqft", "Coats",
        "Labor_Hours", "Distance", "Concurrent_Job", "GP_Percent"
    }
    if not required.issubset(raw.columns):
        st.error(f"Missing columns: {required - set(raw.columns)}")
        return None, None

    raw = raw[list(required)].dropna()
    raw["Quoted_Price_per_sqft"] = raw["Quoted_Price"] / raw["Size_sqft"]

    X_gp = raw[[
        "Quoted_Price_per_sqft",
        "Size_sqft",
        "Coats",
        "Labor_Hours",
        "Distance",
        "Concurrent_Job"
    ]]
    y_gp = raw["GP_Percent"]
    gp_model = RandomForestRegressor().fit(X_gp, y_gp)

    return labor_model, gp_model


# -----------------------------------------------------------------------------
# 3. INITIALIZE
# -----------------------------------------------------------------------------
labor_model, gp_model = train_models()
if labor_model is None or gp_model is None:
    st.stop()

# -----------------------------------------------------------------------------
# 4. APP UI
# -----------------------------------------------------------------------------
logo = Image.open("image.png")
st.image(logo, width=300)
st.title("Gym Floor Pricing Estimator")

square_feet       = st.number_input("Square footage", min_value=4000, max_value=100000, step=100)
coats             = st.selectbox("Number of coats", options=[1, 2])
distance          = st.number_input("Distance to job site (miles)", min_value=1, max_value=500)
concurrent        = st.selectbox("Concurrent job?", options=["No", "Yes"])
concurrent_flag   = 1 if concurrent == "Yes" else 0

if st.button("Estimate Price"):
    # Step 1: Predict labor hours
    inp = [[square_feet, coats, distance, concurrent_flag]]
    predicted_labor = float(labor_model.predict(inp)[0])

    # Step 2: Loop to satisfy GP% ≥ 45% & floor pricing
    floor_price = 0.67 if coats == 2 else 0.48
    best_ppsf   = floor_price
    for ppsf in np.arange(floor_price, 2.01, 0.01):
        gp_in = [[
            ppsf,
            square_feet,
            coats,
            predicted_labor,
            distance,
            concurrent_flag
        ]]
        if gp_model.predict(gp_in)[0] >= 45.0:
            best_ppsf = ppsf
            break

    total_estimate = best_ppsf * square_feet

    # Step 3: Display & log
    st.subheader("Estimate")
    st.write(f"Predicted labor hours: {predicted_labor:.1f}")
    st.write(f"Price per sq ft:       ${best_ppsf:.2f}")
    st.write(f"Total estimated price: ${total_estimate:,.2f}")

    log_prediction(
        square_feet,
        coats,
        distance,
        concurrent_flag,
        predicted_labor,
        best_ppsf,
        total_estimate
    )