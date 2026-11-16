
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
result_folder = PROJECT_ROOT / "result/stage_4_result"
result_folder.mkdir(parents=True, exist_ok=True)

# Training results from our completed experiments
classification = {
    "RNN": {
        "loss": [0.6919, 0.6566, 0.6186, 0.6038, 0.6252],
        "accuracy": [53.51, 60.73, 65.90, 66.90, 64.23]
    },
    "LSTM": {
        "loss": [0.6429, 0.5504, 0.5666, 0.5271, 0.3714],
        "accuracy": [62.95, 73.09, 70.62, 74.02, 83.98]
    },
    "GRU": {
        "loss": [0.6272, 0.4520, 0.3090, 0.2264, 0.1656],
        "accuracy": [63.93, 79.70, 87.26, 91.44, 94.02]
    }
}

generation = {
    "RNN": {
        "loss": [
            5.8474, 4.7941, 4.2947, 3.8920, 3.5410,
            3.2238, 2.9385, 2.6813, 2.4537, 2.2482,
            2.0659, 1.9020, 1.7533, 1.6201, 1.4973,
            1.3867, 1.2858, 1.1941, 1.1103, 1.0329
        ],
        "accuracy": [
            17.18, 24.84, 27.64, 30.24, 32.69,
            35.54, 39.35, 43.48, 47.25, 50.99,
            54.60, 57.82, 60.72, 63.54, 66.33,
            68.67, 71.01, 73.26, 75.03, 76.92
        ]
    },
    "LSTM": {
        "loss": [
            5.9625, 4.9743, 4.4814, 4.0718, 3.7066,
            3.3747, 3.0693, 2.7830, 2.5170, 2.2709,
            2.0474, 1.8476, 1.6649, 1.5027, 1.3575,
            1.2259, 1.1087, 1.0035, 0.9102, 0.8275
        ],
        "accuracy": [
            15.85, 24.39, 27.92, 30.47, 33.14,
            35.68, 38.64, 42.26, 46.52, 51.44,
            56.08, 60.29, 64.42, 67.98, 71.17,
            74.33, 76.91, 79.25, 81.37, 83.13
        ]
    },
    "GRU": {
        "loss": [
            5.8440, 4.7364, 4.1906, 3.7466, 3.3606,
            3.0112, 2.6934, 2.4056, 2.1464, 1.9180,
            1.7148, 1.5326, 1.3731, 1.2297, 1.1033,
            0.9937, 0.8970, 0.8119, 0.7367, 0.6759
        ],
        "accuracy": [
            17.40, 26.19, 29.77, 32.67, 35.53,
            39.06, 43.27, 48.31, 53.43, 57.95,
            62.34, 66.72, 70.26, 73.63, 76.47,
            79.14, 81.35, 83.13, 84.77, 86.14
        ]
    }
}

colors = {
    "RNN": "blue",
    "LSTM": "green",
    "GRU": "red"
}

def plot_combined(results, metric, title, ylabel, filename):
    plt.figure(figsize=(8, 5))

    for model_name, values in results.items():
        epochs = range(1, len(values[metric]) + 1)

        plt.plot(
            epochs,
            values[metric],
            label=model_name,
            color=colors[model_name],
            linewidth=2,
            marker="o",
            markersize=4
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(result_folder / filename, dpi=300)
    plt.close()

# Classification plots
plot_combined(
    classification,
    "loss",
    "IMDb Classification - Training Loss",
    "Cross-Entropy Loss",
    "Classification_Combined_Loss.png"
)

plot_combined(
    classification,
    "accuracy",
    "IMDb Classification - Training Accuracy",
    "Training Accuracy (%)",
    "Classification_Combined_Accuracy.png"
)

# Generation plots
plot_combined(
    generation,
    "loss",
    "Joke Generation - Training Loss",
    "Cross-Entropy Loss",
    "Generation_Combined_Loss.png"
)

plot_combined(
    generation,
    "accuracy",
    "Joke Generation - Next-Word Accuracy",
    "Training Next-Word Accuracy (%)",
    "Generation_Combined_Accuracy.png"
)

print("All four combined convergence plots saved!")
