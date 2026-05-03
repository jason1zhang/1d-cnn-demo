"""
generate_data.py
Creates a synthetic dataset of 10,000 noisy sine‑wave signals (3 classes)
and saves it to synthetic_signals.csv (with sample_id for unique identification).
"""

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Data generation function
# ------------------------------------------------------------
def generate_signals(n_samples=10000, length=100, noise_level=0.3):
    """
    Generate three classes of noisy sine wave signals.
    Class 0: slow wave (base frequency 0.5 Hz)
    Class 1: medium wave (base frequency 1.0 Hz)
    Class 2: fast wave (base frequency 2.0 Hz)
    Each signal has random phase, amplitude scaling, and slight frequency perturbation.
    """
    X, y = [], []
    t = np.linspace(0, 1, length)
    for _ in range(n_samples):
        cls = np.random.randint(0, 3)
        if cls == 0:
            base_freq = 0.5
        elif cls == 1:
            base_freq = 1.0
        else:
            base_freq = 2.0

        freq = base_freq + np.random.uniform(-0.05, 0.05)
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.8, 1.2)

        signal = amp * np.sin(2 * np.pi * freq * t + phase)
        signal += noise_level * np.random.randn(length)
        X.append(signal)
        y.append(cls)

    return np.array(X), np.array(y)

# ------------------------------------------------------------
# Generate and save
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Generating data...")
    X, y = generate_signals(n_samples=10000, length=100, noise_level=0.3)
    print(f"Data shape: X = {X.shape}, y = {y.shape}")

    # Build a DataFrame: first column 'label', then t0, t1, ..., t99
    columns = ['label'] + [f't{i}' for i in range(X.shape[1])]
    data = np.column_stack((y, X))          # shape (10000, 101)
    df = pd.DataFrame(data, columns=columns)
    df['label'] = df['label'].astype(int)   # store labels as integers

    # ---- ADD UNIQUE IDENTIFIER ----
    df.insert(0, 'sample_id', range(len(df)))   # 0, 1, 2, ..., 9999

    df.to_csv('synthetic_signals_single_channel.csv', index=False)
    print("Dataset saved to synthetic_signals_single_channel.csv")
    print("First 5 rows:\n", df.head())