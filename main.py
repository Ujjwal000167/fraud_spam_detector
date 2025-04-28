# src/main.py

import subprocess

def run_train():
    """Train the models and save them."""
    print("\nTraining models...")
    subprocess.run(["python", "src/train_model.py"])

def run_predict():
    """Run the prediction script."""
    print("\nRunning prediction interface...")
    subprocess.run(["python", "src/predict.py"])

def run_evaluate():
    """Evaluate the trained models."""
    print("\nEvaluating models...")
    subprocess.run(["python", "src/evaluate_model.py"])

def main():
    while True:
        print("\nChoose an option:")
        print("1. Train Models")
        print("2. Run Prediction Interface")
        print("3. Evaluate Models")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1/2/3/4): ").strip()
        
        if choice == '1':
            run_train()
        elif choice == '2':
            run_predict()
        elif choice == '3':
            run_evaluate()
        elif choice == '4':
            print("\nExiting... Goodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
