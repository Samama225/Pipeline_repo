import streamlit as st
import pickle
import numpy as np

# Load model
with open("churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

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

# Create input fields dynamically based on features
inputs = []
for feat in features:
    val = st.number_input(f"Enter {feat}", value=0.0)
    inputs.append(val)

if st.button("Predict"):
    sample = np.array([inputs])
    pred = model.predict(sample)[0]

    st.success(f"Predicted next-day consumption (kWh): {pred:.2f}")

    # Example recommendation logic
    if pred < 2:
        plan = "Plan A"
    elif pred < 4:
        plan = "Plan B"
    else:
        plan = "Plan C"

    st.info(f"Recommended Energy Plan: {plan}")
