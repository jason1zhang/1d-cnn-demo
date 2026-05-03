"""
generate_data.py
Creates a synthetic dataset of 10,000 multi-channel (3 channels)
noisy sine-wave signals (3 classes) and saves to synthetic_signals_mc.csv.
Each sample has 3 channels, each of length 100.
Class 0: slow waves, Class 1: medium waves, Class 2: fast waves.
"""

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Multi-channel data generation
# ------------------------------------------------------------
def generate_multichannel_signals(n_samples=10000, length=100, n_channels=3, noise_level=0.3):
    """
    Generate 3 classes of multi-channel signals.
    Each channel is a sine wave with a distinct frequency,
    random phase, amplitude scaling, and slight frequency perturbation.
    Returns X of shape (n_samples, n_channels, length) and y of shape (n_samples,).
    """
    X, y = [], []
    t = np.linspace(0, 1, length)
    # Frequency sets for each class (one per channel)
    class_freqs = {
        0: [0.5, 1.0, 1.5],
        1: [1.0, 2.0, 2.5],
        2: [2.0, 3.0, 3.5]
    }
    for _ in range(n_samples):
        cls = np.random.randint(0, 3)
        sample_channels = []
        for ch_idx in range(n_channels):
            base_freq = class_freqs[cls][ch_idx]
            freq = base_freq + np.random.uniform(-0.05, 0.05)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.8, 1.2)
            signal = amp * np.sin(2 * np.pi * freq * t + phase)
            signal += noise_level * np.random.randn(length)
            sample_channels.append(signal)
        # Stack channels: shape (n_channels, length)
        X.append(np.stack(sample_channels, axis=0))
        y.append(cls)
    return np.array(X), np.array(y)

# ------------------------------------------------------------
# Generate and save
# ------------------------------------------------------------
if __name__ == "__main__":
    n_channels = 3
    length = 100
    print("Generating multi-channel data...")
    X, y = generate_multichannel_signals(n_samples=10000, length=length,
                                         n_channels=n_channels, noise_level=0.3)
    print(f"Data shape: X = {X.shape}, y = {y.shape}")  # (10000, 3, 100)

    # Build columns for CSV: sample_id, label, ch0_t0..ch0_t99, ch1_t0..ch1_t99, ch2_t0..ch2_t99
    columns = ['label']
    for ch in range(n_channels):
        columns += [f'ch{ch}_t{i}' for i in range(length)]

    # Reshape X to (n_samples, n_channels*length) and combine with y
    X_flat = X.reshape(X.shape[0], -1)                 # (10000, 300)
    data = np.column_stack((y, X_flat))                # (10000, 301)
    df = pd.DataFrame(data, columns=columns)
    df['label'] = df['label'].astype(int)

    # Add unique identifier
    df.insert(0, 'sample_id', range(len(df)))

    df.to_csv('synthetic_signals_mc.csv', index=False)
    print("Dataset saved to synthetic_signals_mc.csv")
    print("First 3 rows:\n", df.head(3))