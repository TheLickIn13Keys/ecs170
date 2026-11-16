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

# Load the dataset once
loader = Dataset_Loader_Classification("IMDb", "")
loader.dataset_source_folder_path = str(
    PROJECT_ROOT / "data/stage_4_data/text_classification"
)
data = loader.load()

result_folder = PROJECT_ROOT / "result/stage_4_result"
result_folder.mkdir(parents=True, exist_ok=True)

# Train LSTM and GRU
for rnn_type in ["LSTM", "GRU"]:

    print(f"\n========== {rnn_type} ==========")

    torch.manual_seed(2)
    np.random.seed(2)

    model = Method_RNN_Classification(
        f"{rnn_type}_Classification",
        f"IMDb classification using {rnn_type}",
        vocab_size=data["vocab_size"],
        embedding_dim=100,
        hidden_dim=128,
        rnn_type=rnn_type
    )

    model.data = data
    model.max_epoch = 5
    model.learning_rate = 0.001
    model.batch_size = 64

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

    print(f"\n=== {rnn_type} Results ===")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")

    # Plot training loss
    plt.figure()
    plt.plot(
        range(1, model.max_epoch + 1),
        result["train_loss"],
        marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"{rnn_type} Classification Training Loss")
    plt.grid(True)
    plt.savefig(
        result_folder / f"{rnn_type}_Classification_loss.png"
    )
    plt.close()

    # Plot training accuracy
    plt.figure()
    plt.plot(
        range(1, model.max_epoch + 1),
        np.array(result["train_acc"]) * 100,
        marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy (%)")
    plt.title(f"{rnn_type} Classification Training Accuracy")
    plt.grid(True)
    plt.savefig(
        result_folder / f"{rnn_type}_Classification_accuracy.png"
    )
    plt.close()

print("\nFinished LSTM and GRU experiments!")
