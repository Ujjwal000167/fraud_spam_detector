# src/evaluate_model.py

import joblib  # Correct import for joblib
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load datasets with encoding specified
fraud_data = pd.read_csv('data/creditcard.csv')
spam_data = pd.read_csv('data/spam.csv', encoding='ISO-8859-1')  # Try 'ISO-8859-1' or 'latin1'

# Check the first few rows and columns to inspect structure
print(spam_data.head())  # Print the first few rows to inspect the data

# --- Preprocessing Spam Dataset ---
# Find the correct column name for labels (e.g., it might be 'v1' or 'v2')
spam_data['Category'] = spam_data['v1'].map({'ham': 0, 'spam': 1})  # Assuming 'v1' contains the labels

# Continue with other processing steps (like vectorizing)
vectorizer = joblib.load('models/spam_vectorizer.pkl')
X_spam = vectorizer.transform(spam_data['v2']).toarray()  # Assuming 'v2' contains the message text
y_spam = spam_data['Category']

# Load models
fraud_model = joblib.load('models/fraud_model.pkl')
spam_model = joblib.load('models/spam_model.pkl')

# --- Evaluating Fraud Model ---
print("\nEvaluating Credit Card Fraud Model...")
fraud_predictions = fraud_model.predict(fraud_data.drop('Class', axis=1))
fraud_accuracy = accuracy_score(fraud_data['Class'], fraud_predictions)
fraud_report = classification_report(fraud_data['Class'], fraud_predictions)
print(f"Fraud Model Accuracy: {fraud_accuracy}")
print(fraud_report)

# --- Evaluating Spam Model ---
print("\nEvaluating SMS Spam Model...")
spam_predictions = spam_model.predict(X_spam)
spam_accuracy = accuracy_score(y_spam, spam_predictions)
spam_report = classification_report(y_spam, spam_predictions)
print(f"Spam Model Accuracy: {spam_accuracy}")
print(spam_report)
