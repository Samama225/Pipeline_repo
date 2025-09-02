import pickle
import pandas as pd

def test_model():
    # Load the trained models (dictionary with regressor + classifier)
    with open("churn_model.pkl", "rb") as f:
        models = pickle.load(f)

    regressor = models["regressor"]
    classifier = models["classifier"]

    # Create a dummy input with the required features
    sample = pd.DataFrame([{
        "Global_reactive_power": 0.3,
        "Voltage": 240.0,
        "Global_intensity": 12.0,
        "Sub_metering_1": 1.2,
        "Sub_metering_2": 0.5,
        "Sub_metering_3": 0.3
    }])

    # Run predictions separately
    pred_reg = regressor.predict(sample)
    pred_clf = classifier.predict(sample)

    # Assertions
    assert pred_reg is not None
    assert pred_clf is not None

    print("Model test passed.")
    print("Predicted next-day consumption (kWh):", pred_reg[0])
    print("Recommended energy plan:", pred_clf[0])

if __name__ == "__main__":
    test_model()
