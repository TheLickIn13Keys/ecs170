from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_4_code.Dataset_Loader_Classification import (
    Dataset_Loader_Classification
)
from local_code.stage_4_code.Method_RNN_Classification import (
    Method_RNN_Classification
)

# Set random seeds
torch.manual_seed(2)
np.random.seed(2)

# Load IMDb dataset
loader = Dataset_Loader_Classification("IMDb", "")
loader.dataset_source_folder_path = str(
    PROJECT_ROOT / "data/stage_4_data/text_classification"
)

data = loader.load()

# Create RNN model
model = Method_RNN_Classification(
    "RNN_Classification",
    "IMDb sentiment classification using vanilla RNN",
    vocab_size=data["vocab_size"],
    embedding_dim=100,
    hidden_dim=128
)

model.data = data
model.max_epoch = 5
model.learning_rate = 0.001
model.batch_size = 64

# Train and test
result = model.run()

y_true = result["true_y"]
y_pred = result["pred_y"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)
recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)
f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

print("\n=== RNN Classification Results ===")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-score:  {f1 * 100:.2f}%")

# Save learning curves
result_folder = PROJECT_ROOT / "result/stage_4_result"
result_folder.mkdir(parents=True, exist_ok=True)

plt.figure()
plt.plot(
    range(1, model.max_epoch + 1),
    result["train_loss"],
    marker="o"
)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("IMDb RNN Training Loss")
plt.grid(True)
plt.savefig(result_folder / "RNN_Classification_loss.png")
plt.close()

plt.figure()
plt.plot(
    range(1, model.max_epoch + 1),
    np.array(result["train_acc"]) * 100,
    marker="o"
)
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy (%)")
plt.title("IMDb RNN Training Accuracy")
plt.grid(True)
plt.savefig(result_folder / "RNN_Classification_accuracy.png")
plt.close()

print("\nLearning curves saved.")
