# src/train_models.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# --- Create models/ directory if not exists ---
os.makedirs('models', exist_ok=True)

# --- Load datasets ---
print("\nLoading datasets...")
fraud_data = pd.read_csv('data/creditcard.csv')
spam_data = pd.read_csv('data/spam.csv', encoding='latin1')

# --- Preprocess Credit Card Fraud Dataset ---
print("\nPreprocessing Credit Card Fraud Dataset...")

X_fraud = fraud_data.drop('Class', axis=1)
y_fraud = fraud_data['Class']

scaler = StandardScaler()
X_fraud_scaled = scaler.fit_transform(X_fraud)

# --- Preprocess SMS Spam Dataset ---
print("\nPreprocessing SMS Spam Dataset...")

spam_data.rename(columns={spam_data.columns[0]: "label", spam_data.columns[1]: "message"}, inplace=True)
spam_data['label'] = spam_data['label'].map({'ham': 0, 'spam': 1})

vectorizer = TfidfVectorizer()
X_spam = vectorizer.fit_transform(spam_data['message']).toarray()
y_spam = spam_data['label']

# --- Train Models ---
print("\nTraining Models...")

fraud_model = LogisticRegression(max_iter=1000)
fraud_model.fit(X_fraud_scaled, y_fraud)

spam_model = LogisticRegression(max_iter=1000)
spam_model.fit(X_spam, y_spam)

# --- Save Models ---
print("\nSaving Models...")

joblib.dump(fraud_model, 'models/fraud_model.pkl')
joblib.dump(spam_model, 'models/spam_model.pkl')
joblib.dump(scaler, 'models/fraud_scaler.pkl')
joblib.dump(vectorizer, 'models/spam_vectorizer.pkl')

print("\nModels saved successfully!")
