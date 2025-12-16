import pandas as pd
from scipy.stats import ks_2samp

def check_drift():
    print("--- Model Drift Analysis ---")
    try:
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")
    except:
        print("Error: Could not find train.csv or test.csv")
        return

    # Columns to check for drift
    features = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
    
    print(f"{'Feature':<15} | {'P-Value':<10} | {'Drift Detected?'}")
    print("-" * 50)
    
    for col in features:
        # We use the KS Test to compare distributions
        # If p-value < 0.05, it means the data looks SIGNIFICANTLY different (Drift)
        stat, p_value = ks_2samp(train[col], test[col])
        is_drift = "YES" if p_value < 0.05 else "NO"
        
        print(f"{col:<15} | {p_value:.5f}    | {is_drift}")

    print("\nAnalysis Complete.")

if __name__ == "__main__":
    check_drift()

