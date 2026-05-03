# 1D-CNN Demo – Multi-Channel Synthetic Signal Classification

A clean, self-contained PyTorch project that demonstrates how to build and train a **1‑Dimensional Convolutional Neural Network (1D‑CNN)** for multi‑channel time‑series classification.

Everything is based on **synthetic data** – no external downloads required. The scripts generate three classes of noisy, multi‑channel sine waves, save them to a CSV for manual inspection, then train a small CNN to classify them.  
The entire pipeline can be easily adapted to real sensor data (ECG, accelerometers, vibration monitoring, etc.).

---

## 📁 Project Structure
1d-cnn-demo/
├── generate_data.py # Create synthetic multi-channel signals & export to CSV
├── train_model.py # Load CSV, train 1D-CNN, evaluate, plot results
└── README.md

After running the scripts you will also see:
- `synthetic_signals_mc.csv` – the generated dataset (can be inspected in Excel)
- `training_curves_mc.png` – plots of loss & accuracy during training

---

## 🧠 What the Model Learns

The synthetic dataset contains **10,000 samples**, each with:
- **3 channels** (simulating e.g. a tri‑axial accelerometer)
- **100 time steps** (1 second sampled at 100 Hz)
- **3 classes** (slow, medium, fast oscillations)

The 1D CNN has three convolutional layers, global average pooling, and a dense classifier.  
It learns to distinguish the frequency patterns **directly from raw waveforms**, even with noise, random phases, and amplitude jitter.

---

## Model Architecture (simplified)
Input: (batch, 3, 100)
  Conv1D (3 → 16, kernel 5, 'same')
  ReLU + MaxPool1D(2)           → (batch, 16, 50)
  Conv1D (16 → 32, kernel 5, 'same')
  ReLU + MaxPool1D(2)           → (batch, 32, 25)
  Conv1D (32 → 64, kernel 5, 'same')
  ReLU                          → (batch, 64, 25)
  GlobalAveragePool1D           → (batch, 64, 1)
  Squeeze + Dense(64) + ReLU + Dropout(0.5)
  Dense(3)                      → (batch, 3)


---

## License
This project is provided as a learning resource. Feel free to use, modify, and share it as you wish.
