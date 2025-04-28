# src/load_data.py

import pandas as pd

# --- Load Credit Card Fraud Dataset ---
print("\nLoading Credit Card Fraud Dataset...")
fraud_data = pd.read_csv('data/creditcard.csv')

print("\nFirst 5 rows of Fraud Dataset:")
print(fraud_data.head())

print("\nFraud Dataset Info:")
print(fraud_data.info())

print("\nMissing values in Fraud Dataset:")
print(fraud_data.isnull().sum())

# --- Load SMS Spam Dataset ---
print("\nLoading SMS Spam Dataset...")
spam_data = pd.read_csv('data/spam.csv', encoding='latin-1')

print("\nFirst 5 rows of Spam Dataset:")
print(spam_data.head())

print("\nSpam Dataset Info:")
print(spam_data.info())

print("\nMissing values in Spam Dataset:")
print(spam_data.isnull().sum())
