import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np

# Load models and preprocessors
fraud_model = joblib.load('models/fraud_model.pkl')
spam_model = joblib.load('models/spam_model.pkl')
fraud_scaler = joblib.load('models/fraud_scaler.pkl')
spam_vectorizer = joblib.load('models/spam_vectorizer.pkl')

# Create the main window
class FraudSpamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fraud and Spam Detection")
        self.root.geometry("500x500")
        self.root.config(bg="#f0f0f0")

        # Add a title label
        self.title_label = ttk.Label(self.root, text="Fraud and Spam Detection", font=("Arial", 16, "bold"), background="#f0f0f0")
        self.title_label.pack(pady=20)

        # Frame for Fraud Prediction
        self.fraud_frame = ttk.Frame(self.root, padding="10")
        self.fraud_frame.pack(pady=20)

        self.fraud_label = ttk.Label(self.fraud_frame, text="Fraud Detection", font=("Arial", 14, "bold"))
        self.fraud_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.feature1_label = ttk.Label(self.fraud_frame, text="Feature 1:")
        self.feature1_label.grid(row=1, column=0, sticky="w", padx=10)
        self.feature1_input = ttk.Entry(self.fraud_frame)
        self.feature1_input.grid(row=1, column=1, padx=10)

        self.feature2_label = ttk.Label(self.fraud_frame, text="Feature 2:")
        self.feature2_label.grid(row=2, column=0, sticky="w", padx=10)
        self.feature2_input = ttk.Entry(self.fraud_frame)
        self.feature2_input.grid(row=2, column=1, padx=10)

        self.fraud_predict_button = ttk.Button(self.fraud_frame, text="Predict Fraud", command=self.predict_fraud)
        self.fraud_predict_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.fraud_result_label = ttk.Label(self.fraud_frame, text="Fraud Prediction: ", font=("Arial", 12))
        self.fraud_result_label.grid(row=4, column=0, columnspan=2, pady=10)

        # Frame for Spam Prediction
        self.spam_frame = ttk.Frame(self.root, padding="10")
        self.spam_frame.pack(pady=20)

        self.spam_label = ttk.Label(self.spam_frame, text="Spam Detection", font=("Arial", 14, "bold"))
        self.spam_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.spam_message_label = ttk.Label(self.spam_frame, text="Enter Message:")
        self.spam_message_label.grid(row=1, column=0, sticky="w", padx=10)
        self.spam_message_input = tk.Text(self.spam_frame, height=4, width=30)
        self.spam_message_input.grid(row=1, column=1, padx=10)

        self.spam_predict_button = ttk.Button(self.spam_frame, text="Predict Spam", command=self.predict_spam)
        self.spam_predict_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.spam_result_label = ttk.Label(self.spam_frame, text="Spam Prediction: ", font=("Arial", 12))
        self.spam_result_label.grid(row=3, column=0, columnspan=2, pady=10)

    # Method for predicting fraud
    def predict_fraud(self):
        try:
            feature1 = float(self.feature1_input.get())
            feature2 = float(self.feature2_input.get())

            # Check if the inputs are valid
            if feature1 < 0 or feature2 < 0:
                messagebox.showerror("Input Error", "Please enter positive values for features.")
                return

            features = np.array([feature1, feature2]).reshape(1, -1)
            scaled_features = fraud_scaler.transform(features)
            prediction = fraud_model.predict(scaled_features)

            if prediction == 1:
                self.fraud_result_label.config(text="Fraud Prediction: Fraud", foreground="red")
            else:
                self.fraud_result_label.config(text="Fraud Prediction: Not Fraud", foreground="green")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values for features.")

    # Method for predicting spam
    def predict_spam(self):
        message = self.spam_message_input.get("1.0", "end-1c").strip()

        if message:
            message_vectorized = spam_vectorizer.transform([message])
            prediction = spam_model.predict(message_vectorized)

            if prediction == 1:
                self.spam_result_label.config(text="Spam Prediction: Spam", foreground="red")
            else:
                self.spam_result_label.config(text="Spam Prediction: Not Spam", foreground="green")
        else:
            messagebox.showerror("Input Error", "Please enter a message to check for spam.")

# Initialize the Tkinter root window
if __name__ == '__main__':
    root = tk.Tk()
    app = FraudSpamApp(root)
    root.mainloop()
