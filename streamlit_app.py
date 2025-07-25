import os
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------------------------
# A) LOGGING PREDICTIONS
# -----------------------------------------------------------------------------
def log_prediction(job_id, sqft, coats, dist, conc, pred_hours, ppsf, total):
    logfile = "data/prediction_log.csv"
    first = not os.path.isfile(logfile)
    row = {
        "timestamp": datetime.now().isoformat(),
        "job_id": job_id,
        "Size_sqft": sqft,
        "Coats": coats,
        "Distance": dist,
        "Concurrent_Job": conc,
        "Predicted_Labor_Hours": round(pred_hours,2),
        "Price_per_sqft": round(ppsf,2),
        "Total_Price": round(total,2)
    }
    pd.DataFrame([row]).to_csv(logfile, mode="a", index=False, header=first)

# -----------------------------------------------------------------------------
# B) TRAINING MODELS (cached)
# -----------------------------------------------------------------------------
@st.cache_resource(ttl=3600)
def train_models():
    # 1) LinearRegression on your ground_truth file
    gt = pd.read_csv("data/ground_truth.csv")
    gt = gt.dropna(subset=[
        "Size_sqft","Coats","Distance","Concurrent_Job","Actual_Labor_Hours"
    ])
    X_lab = gt[["Size_sqft","Coats","Distance","Concurrent_Job"]]
    y_lab = gt["Actual_Labor_Hours"]
    labor_model = LinearRegression().fit(X_lab, y_lab)

    # 2) RandomForestRegressor for GP%
    rf_df = pd.read_csv("data/gym_floor_raw.csv")
    # … your existing gym_floor preprocessing …
    rf_df["Quoted_Price_per_sqft"] = rf_df["Quoted_Price"] / rf_df["Size_sqft"]
    X_gp = rf_df[[
        "Quoted_Price_per_sqft",
        "Size_sqft","Coats","Labor_Hours","Distance","Concurrent_Job"
    ]]
    y_gp = rf_df["GP_Percent"]
    gp_model = RandomForestRegressor().fit(X_gp, y_gp)

    return labor_model, gp_model

# -----------------------------------------------------------------------------
# C) STREAMLIT UI
# -----------------------------------------------------------------------------
labor_model, gp_model = train_models()

st.title("Gym Floor Estimator")

# generate or input a job_id
job_id = st.text_input("Job ID", value=str(int(datetime.now().timestamp())))

sqft  = st.number_input("Square footage", value=5000, step=100)
coats = st.selectbox("Coats", [1,2])
dist  = st.number_input("Distance (miles)", value=5, step=1)
conc  = st.selectbox("Concurrent job?", ["No","Yes"])
conc_flag = 1 if conc=="Yes" else 0

if st.button("Estimate"):
    # 1) Predict labor hours
    inp = [[sqft, coats, dist, conc_flag]]
    ph = float(labor_model.predict(inp)[0])

    # 2) Find price‐per‐sqft for ≥45% GP
    floor_ppsf = 0.67 if coats==2 else 0.48
    best_ppsf = floor_ppsf
    for p in [floor_ppsf + 0.01*i for i in range(0,200)]:
        gp_in = [[p, sqft, coats, ph, dist, conc_flag]]
        if gp_model.predict(gp_in)[0] >= 45.0:
            best_ppsf = round(p,2)
            break

    total = best_ppsf * sqft

    # 3) Display & log
    st.write(f"Pred labor hrs: {ph:.1f}")
    st.write(f"Price/sqft:  ${best_ppsf:.2f}")
    st.write(f"Total est.:  ${total:,.2f}")

    log_prediction(job_id, sqft, coats, dist, conc_flag, ph, best_ppsf, total)
    push_log_to_github()

import base64
import requests

def push_log_to_github():
    with open("labor_predictions_log.csv", "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    url = f"https://api.github.com/repos/{st.secrets['github']['repo']}/contents/labor_predictions_log.csv"
    headers = {
        "Authorization": f"Bearer {st.secrets['github']['token']}",
        "Accept": "application/vnd.github+json"
    }

    # Check if file already exists (needed to get SHA for overwrite)
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        sha = resp.json()["sha"]
        payload = {
            "message": "Update log file",
            "content": encoded,
            "sha": sha,
            "branch": st.secrets["github"]["branch"]
        }
    else:
        payload = {
            "message": "Create log file",
            "content": encoded,
            "branch": st.secrets["github"]["branch"]
        }

    push = requests.put(url, headers=headers, json=payload)
    st.success("Logs pushed to GitHub securely.")