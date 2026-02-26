import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def train_model():

    df = pd.read_csv("data/churn.csv")

    # Drop customer ID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Convert target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Remove missing values
    df = df.dropna()

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Save column structure
    os.makedirs("model", exist_ok=True)
    joblib.dump(X.columns.tolist(), "model/columns.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create Pipeline (Scaler + Model)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "model/logistic_model.pkl")

    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()