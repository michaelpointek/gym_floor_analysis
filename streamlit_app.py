import os
import math
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------------
# A) LOGGING PREDICTIONS
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# B) TRAINING MODELS (cached)
# ---------------------------------------------------------------------
@st.cache_resource(ttl=3600)
def train_models():
    gt = pd.read_csv("data/ground_truth.csv").dropna(subset=[
        "Size_sqft", "Coats", "Distance", "Concurrent_Job", "Actual_Labor_Hours", "v_mat_cost"
    ])
    avg_vmat_by_coat = (
        gt
        .assign(vmat_per_sqft = gt["v_mat_cost"] / gt["Size_sqft"])
        .groupby("Coats")["vmat_per_sqft"]
        .mean()
        .to_dict()
    )
    X_lab = gt[["Size_sqft","Coats","Distance","Concurrent_Job"]]
    y_lab = gt["Actual_Labor_Hours"]
    labor_model = LinearRegression().fit(X_lab, y_lab)

    rf_df = pd.read_csv("data/gym_floor_raw.csv").dropna(subset=[
        "Quoted_Price", "Size_sqft", "Coats", "Labor_Hours",
        "Distance", "Concurrent_Job", "Labor_Cost", "v_mat_cost"
    ])
    rf_df["Quoted_Price_per_sqft"] = rf_df["Quoted_Price"] / rf_df["Size_sqft"]
    rf_df["Total_Cost"] = rf_df["Labor_Cost"] + rf_df["v_mat_cost"]
    rf_df["GP_Percent"] = (rf_df["Quoted_Price"] - rf_df["Total_Cost"]) / rf_df["Quoted_Price"] * 100

    X_gp = rf_df[[
        "Quoted_Price_per_sqft", "Size_sqft", "Coats", "Labor_Hours",
        "Distance", "Concurrent_Job", "Labor_Cost", "v_mat_cost"
    ]]
    y_gp = rf_df["GP_Percent"]
    gp_model = RandomForestRegressor().fit(X_gp, y_gp)

    return labor_model, gp_model

# ---------------------------------------------------------------------
# C) REVERSE GP% PRICING FUNCTION
# ---------------------------------------------------------------------
def compute_target_ppsf(sqft, coats, labor_hours, target_gp, avg_vmat_by_coat):
    labor_cost = labor_hours * 19.4
    v_mat_rate = avg_vmat_by_coat.get(coats, 0)  # fallback to 0 if coats not found
    v_mat_cost = v_mat_rate * sqft
    total_cost = labor_cost + v_mat_cost
    target_price = total_cost / (1 - target_gp)
    target_ppsf = math.ceil((target_price / sqft) * 100) / 100
    return target_ppsf, round(target_price, 2)

# ---------------------------------------------------------------------
# D) STREAMLIT UI
# ---------------------------------------------------------------------
labor_model, gp_model = train_models()

st.title("Gym Floor Estimator")

job_id = st.text_input("Job ID", value=str(int(datetime.now().timestamp())))
sqft  = st.number_input("Square footage", value=5000, step=100)
coats = st.selectbox("Coats", [1,2])
dist  = st.number_input("Distance (miles)", value=5, step=1)
conc  = st.selectbox("Concurrent job?", ["No","Yes"])
conc_flag = 1 if conc=="Yes" else 0

if st.button("Estimate"):
    inp = [[sqft, coats, dist, conc_flag]]
    ph = float(labor_model.predict(inp)[0])

    # Reverse GP% logic
    target_gp = 0.45
    best_ppsf, total = compute_target_ppsf(sqft, coats, ph, target_gp)

    st.write(f"Pred labor hrs: {ph:.1f}")
    st.write(f"Target GP%: {target_gp*100:.0f}%")
    st.write(f"Price/sqft:  ${best_ppsf:.2f}")
    st.write(f"Total est.:  ${best_ppsf * sqft:,.2f}")
    
    with st.expander("Show calculation breakdown"):
        st.write(f"Labor hours: {ph:.2f}")
        st.write(f"Target GP%: {target_gp:.2%}")
        st.write(f"Price per sqft: ${best_ppsf:.2f}")
        st.write(f"Square footage: {sqft:,}")
        st.write(f"Total price: ${best_ppsf * sqft:,.2f}")
        
    log_prediction(job_id, sqft, coats, dist, conc_flag, ph, best_ppsf, total)

# ---------------------------------------------------------------------
# E) ADMIN PANEL
# ---------------------------------------------------------------------
import base64
import requests
from pathlib import Path

def push_log_to_github():
    with open("data/labor_predictions_log.csv", "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    url = f"https://api.github.com/repos/{st.secrets['github']['repo']}/contents/labor_predictions_log.csv"
    headers = {
        "Authorization": f"Bearer {st.secrets['github']['token']}",
        "Accept": "application/vnd.github+json"
    }

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

    job_statuses = df_log['Concurrent_Job'].unique()
    selected_status = st.selectbox("Filter by Concurrent Job", job_statuses)
    filtered = df_log[df_log['Concurrent_Job'] == selected_status]

    st.write("Filtered Log Entries")
    st.dataframe(filtered)

    st.write("📈 Summary Statistics")
    st.write(filtered.describe())

display_admin_panel()