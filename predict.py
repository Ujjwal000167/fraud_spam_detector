# src/predict.py

import pandas as pd
import joblib
import numpy as np

# --- Load Models and Transformers ---
print("\nLoading models and transformers...")
fraud_model = joblib.load('models/fraud_model.pkl')
spam_model = joblib.load('models/spam_model.pkl')
fraud_scaler = joblib.load('models/fraud_scaler.pkl')
spam_vectorizer = joblib.load('models/spam_vectorizer.pkl')

print("\nModels loaded successfully!")

# --- Helper Functions ---
def predict_fraud(features):
    features_scaled = fraud_scaler.transform([features])
    prediction = fraud_model.predict(features_scaled)[0]
    return "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"

def predict_spam(message):
    message_transformed = spam_vectorizer.transform([message]).toarray()
    prediction = spam_model.predict(message_transformed)[0]
    return "Spam Message" if prediction == 1 else "Ham (Not Spam) Message"

# --- Main Program ---
while True:
    print("\nChoose Prediction Type:")
    print("1. Credit Card Fraud Detection")
    print("2. SMS Spam Detection")
    print("3. Exit")
    choice = input("\nEnter your choice (1/2/3): ").strip()

    if choice == '1':
        print("\nEnter 30 features for credit card transaction separated by commas:")
        input_features = input()
        try:
            features = list(map(float, input_features.split(',')))
            if len(features) != 30:
                print("\nError: You must enter exactly 30 features!")
            else:
                result = predict_fraud(features)
                print(f"\nPrediction Result: {result}")
        except:
            print("\nInvalid input. Please enter numbers separated by commas.")

    elif choice == '2':
        print("\nEnter the text message:")
        message = input()
        result = predict_spam(message)
        print(f"\nPrediction Result: {result}")

    elif choice == '3':
        print("\nExiting... Goodbye!")
        break

    else:
        print("\nInvalid choice. Please try again.")
