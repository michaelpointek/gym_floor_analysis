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
    logfile = "data/labor_predictions_log.csv"
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

    rf_df = pd.read_csv("data/gym_floor_raw.csv")

    # Ensure no missing values
    rf_df = rf_df.dropna(subset=[
        "Quoted_Price", "Size_sqft", "Coats", "Labor_Hours",
        "Distance", "Concurrent_Job", "Labor_Cost", "v_mat_cost"
    ])

    # Engineering new fields
    rf_df["Quoted_Price_per_sqft"] = rf_df["Quoted_Price"] / rf_df["Size_sqft"]
    rf_df["Quoted_Price_total"] = rf_df["Quoted_Price"]  # optional alias
    rf_df["Total_Cost"] = rf_df["Labor_Cost"] + rf_df["v_mat_cost"]
    rf_df["GP_Percent"] = (rf_df["Quoted_Price_total"] - rf_df["Total_Cost"]) / rf_df["Quoted_Price_total"] * 100

    X_gp = rf_df[[ 
    "Quoted_Price_per_sqft",
    "Size_sqft", "Coats", "Labor_Hours", 
    "Distance", "Concurrent_Job", "Labor_Cost", "v_mat_cost"
    ]]
    X_gp = rf_df[[ 
        "Quoted_Price_per_sqft",
        "Size_sqft", "Coats", "Labor_Hours", 
        "Distance", "Concurrent_Job", "Labor_Cost", "v_mat_cost"
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
    labor_cost = ph * 19.4
    v_mat_cost = 0.46 * (sqft * floor_ppsf)
    for p in [floor_ppsf + 0.01*i for i in range(0,200)]:
        gp_in = [[p, sqft, coats, ph, dist, conc_flag, labor_cost, v_mat_cost]]
        if gp_model.predict(gp_in)[0] >= 45.0:
            best_ppsf = round(p,2)
            break

    total = best_ppsf * sqft

    # 3) Display & log
    st.write(f"Pred labor hrs: {ph:.1f}")
    st.write(f"Price/sqft:  ${best_ppsf:.2f}")
    st.write(f"Total est.:  ${total:,.2f}")

    log_prediction(job_id, sqft, coats, dist, conc_flag, ph, best_ppsf, total)

    # 4) GP sweep chart
    prices = [floor_ppsf + 0.01*i for i in range(0,200)]
    gps = []
    
    for p in prices:
        labor_cost = ph * 19.4
        if coats == 1:
            v_mat_cost = 0.1981171 * sqft
        else:
            v_mat_cost = 0.3175879 * sqft
    
        gp_val = gp_model.predict([[p, sqft, coats, ph, dist, conc_flag, labor_cost, v_mat_cost]])[0]
        gps.append(gp_val)
    
        st.write(f"Trying ${p:.2f} → GP%: {gp_val:.2f}")
        if gp_val >= 45.0:
            best_ppsf = round(p, 2)
            break


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
    push_log_to_github()

import pandas as pd
import streamlit as st
from pathlib import Path

def load_prediction_log(log_path='data/labor_predictions_log.csv'):
    if Path(log_path).exists():
        return pd.read_csv(log_path)
    else:
        st.warning("Log file not found. Make sure it’s being synced correctly.")
        return pd.DataFrame()

def display_admin_panel():
    st.subheader("🔒 Admin Panel")

    access_key = st.text_input("Enter admin passcode", type="password")
    if access_key == st.secrets["admin"]["passcode"]:
        st.success("Access granted.")
        display_log_panel()
    else:
        st.info("Enter passcode to view logs.")

def display_log_panel():
    st.subheader("📊 Prediction Log Viewer")
    df_log = load_prediction_log()
    
    if df_log.empty:
        st.info("No log data available.")
        return
    
    # Filter by job status or coat count if useful
    job_statuses = df_log['Concurrent_Job'].unique()
    selected_status = st.selectbox("Filter by Concurrent Job", job_statuses)
    filtered = df_log[df_log['Concurrent_Job'] == selected_status]

    st.write("Filtered Log Entries")
    st.dataframe(filtered)

    # Optional: Add summary stats
    st.write("📈 Summary Statistics")
    st.write(filtered.describe())

display_admin_panel()