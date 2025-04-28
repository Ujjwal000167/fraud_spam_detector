# src/preprocess_data.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- Load Credit Card Fraud Dataset ---
print("\nLoading Credit Card Fraud Dataset...")
fraud_data = pd.read_csv('data/creditcard.csv')

# --- Preprocessing Credit Card Fraud Dataset ---
print("\nPreprocessing Credit Card Fraud Dataset...")

# Drop the 'Time' column
fraud_data = fraud_data.drop(['Time'], axis=1)

# Scale the 'Amount' column
scaler = StandardScaler()
fraud_data['Amount'] = scaler.fit_transform(fraud_data[['Amount']])

print("\nFraud dataset after preprocessing:")
print(fraud_data.head())

# --- Load SMS Spam Dataset ---
print("\nLoading SMS Spam Dataset...")
spam_data = pd.read_csv('data/spam.csv', encoding='latin-1')

# --- Preprocessing SMS Spam Dataset ---
print("\nPreprocessing SMS Spam Dataset...")

# Keep only the first two columns (v1 = label, v2 = message)
spam_data = spam_data.iloc[:, :2]
spam_data.columns = ['label', 'message']

print("\nSpam dataset after preprocessing:")
print(spam_data.head())

# Save the cleaned datasets
fraud_data.to_csv('data/clean_fraud.csv', index=False)
spam_data.to_csv('data/clean_spam.csv', index=False)

print("\nCleaned datasets saved as 'clean_fraud.csv' and 'clean_spam.csv'")
