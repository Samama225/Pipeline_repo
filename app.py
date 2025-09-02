import streamlit as st
import pickle
import numpy as np

# Load model dict
with open("churn_model.pkl", "rb") as f:
    data = pickle.load(f)

regressor = data["regressor"]
classifier = data["classifier"]

features = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

st.title("Energy Consumption Prediction App")
st.write("Enter the values below to predict next-day consumption and get a plan recommendation.")

# Input fields
inputs = []
for feat in features:
    val = st.number_input(f"Enter {feat}", value=0.0)
    inputs.append(val)

if st.button("Predict"):
    sample = np.array([inputs])
    
    # Predict consumption
    pred = regressor.predict(sample)[0]
    st.success(f"Predicted next-day consumption (kWh): {pred:.2f}")
    
    # Predict energy plan
    plan = classifier.predict(sample)[0]
    st.info(f"Recommended Energy Plan: {plan}")
