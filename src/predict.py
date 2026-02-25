import joblib
import numpy as np

def predict_new_customer(features):
    model = joblib.load("model/logistic_model.pkl")
    prediction = model.predict([features])

    if prediction[0] == 1:
        return "Customer will Churn"
    else:
        return "Customer will Stay"

if __name__ == "__main__":
    # Example input (must match number of features after preprocessing)
    sample = np.random.rand(1, 10)[0]  # Replace with real feature values
    result = predict_new_customer(sample)
    print(result)