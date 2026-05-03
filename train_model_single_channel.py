"""
train_model.py
Loads synthetic_signals.csv, trains a 1D-CNN classifier, and evaluates.

Expected CSV format:
  label, t0, t1, ..., t99
where each row is a 100‑step signal and its class (0, 1, or 2).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------
# 1. Load dataset from CSV
# ------------------------------------------------------------
print("Loading dataset from synthetic_signals.csv ...")
df = pd.read_csv('synthetic_signals_single_channel.csv')

# Separate labels and features
y = df['label'].values
X = df.drop(columns=['label']).values      # shape (10000, 100)

print(f"Loaded: X = {X.shape}, y = {y.shape}")

# Ensure labels are contiguous 0,1,2 (they already are, but just to be safe)
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split into train/test (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ------------------------------------------------------------
# 2. PyTorch Dataset class
# ------------------------------------------------------------
class SignalDataset(Dataset):
    def __init__(self, signals, labels):
        # Convert to tensor and add channel dimension: (batch, 1, length)
        self.signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

# Create DataLoaders
batch_size = 32
train_dataset = SignalDataset(X_train, y_train)
test_dataset  = SignalDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check one batch shape
for batch_x, batch_y in train_loader:
    print(f"Batch shape: {batch_x.shape}  (expected [batch, 1, 100])")
    break

# ------------------------------------------------------------
# 3. 1D CNN model
# ------------------------------------------------------------
class CNN1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding='same')
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN1D().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------------------------------------------------
# 4. Training with validation
# ------------------------------------------------------------
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
train_sub_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

num_epochs = 50
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("\nStarting training...")
for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for inputs, labels in train_sub_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / val_total
    epoch_val_acc = val_correct / val_total
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

# ------------------------------------------------------------
# 5. Test evaluation
# ------------------------------------------------------------
model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = test_correct / test_total
print(f"\nTest accuracy: {test_acc:.4f}")

# ------------------------------------------------------------
# 6. Plot training curves
# ------------------------------------------------------------
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Validation')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("training_curves_single_channel.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# 7. Single‑sample prediction
# ------------------------------------------------------------
sample_idx = 0
sample_signal, true_label = test_dataset[sample_idx]
sample_signal = sample_signal.unsqueeze(0).to(device)   # add batch dim
model.eval()
with torch.no_grad():
    output = model(sample_signal)
probs = torch.softmax(output, dim=1).cpu().numpy().squeeze()
pred_class = np.argmax(probs)
print(f"\nSingle sample prediction:")
print(f"True class: {true_label.item()}, Predicted class: {pred_class}")
print(f"Class probabilities: {dict(zip(range(3), np.round(probs, 3)))}")