import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and prepare data
@st.cache_resource
@st.cache_resource
def train_model():
    df = pd.read_csv("gym_floor_raw.csv")
    st.write("Column names in CSV:", df.columns.tolist())  # 👈 Shows column names

    # You can rename if the actual columns are slightly off
    df.columns = [col.strip() for col in df.columns]  # Remove extra spaces
    st.write("Cleaned column names:", df.columns.tolist())

    # Uncomment this to view sample data
    # st.dataframe(df.head())

    # Proceed only if required columns exist
    required_columns = {"Type", "Square Feet", "Price"}
    if not required_columns.issubset(set(df.columns)):
        st.error(f"Missing one or more required columns: {required_columns}")
        return None, None

    df = df.dropna(subset=["Type", "Square Feet", "Price"])

    label_enc = LabelEncoder()
    df["Type_encoded"] = label_enc.fit_transform(df["Type"])

    X = df[["Square Feet", "Type_encoded"]]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    return rf, label_enc


# Train once and reuse
model, label_encoder = train_model()

# App UI
st.title("Gym Floor Pricing Estimator")

square_feet = st.number_input("Enter square footage", min_value=100, max_value=10000, step=100)
floor_type = st.selectbox("Select floor type", label_encoder.classes_)

if st.button("Estimate Price"):
    floor_type_encoded = label_encoder.transform([floor_type])[0]
    prediction = model.predict([[square_feet, floor_type_encoded]])
    st.success(f"Estimated price: ${prediction[0]:,.2f}")
