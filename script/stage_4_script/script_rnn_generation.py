from pathlib import Path
import sys
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_4_code.Dataset_Loader_Generation import (
    Dataset_Loader_Generation
)
from local_code.stage_4_code.Method_RNN_Generation import (
    Method_RNN_Generation
)

# Set random seeds
torch.manual_seed(2)
np.random.seed(2)
random.seed(2)

# Load jokes dataset
loader = Dataset_Loader_Generation("Jokes", "")
loader.dataset_source_folder_path = str(
    PROJECT_ROOT / "data/stage_4_data/text_generation/data"
)

data = loader.load()

# Create vanilla RNN generator
model = Method_RNN_Generation(
    "RNN_Generation",
    "Joke generation using vanilla RNN",
    vocab_size=data["vocab_size"],
    embedding_dim=100,
    hidden_dim=128,
    rnn_type="RNN"
)

model.data = data
model.max_epoch = 20
model.learning_rate = 0.001
model.batch_size = 64

# Train
result = model.run()

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
plt.title("RNN Joke Generation Training Loss")
plt.grid(True)
plt.savefig(result_folder / "RNN_Generation_loss.png")
plt.close()

plt.figure()
plt.plot(
    range(1, model.max_epoch + 1),
    np.array(result["train_acc"]) * 100,
    marker="o"
)
plt.xlabel("Epoch")
plt.ylabel("Next-word Accuracy (%)")
plt.title("RNN Joke Generation Training Accuracy")
plt.grid(True)
plt.savefig(result_folder / "RNN_Generation_accuracy.png")
plt.close()

# Generate jokes from three starting words
prompts = [
    "what did the",
    "why did the",
    "how many hipsters"
]

print("\n=== Generated Jokes ===")

for prompt in prompts:
    generated = model.generate(
        prompt,
        data["word_to_id"],
        data["id_to_word"],
        sequence_length=data["sequence_length"],
        max_new_tokens=40
    )

    print(f"\nStarting words: {prompt}")
    print(f"Generated joke: {generated}")

# Save generated jokes to a text file
output_path = result_folder / "RNN_Generation_examples.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for prompt in prompts:
        generated = model.generate(
            prompt,
            data["word_to_id"],
            data["id_to_word"],
            sequence_length=data["sequence_length"],
            max_new_tokens=40
        )

        f.write(f"Starting words: {prompt}\n")
        f.write(f"Generated joke: {generated}\n\n")

print("\nLearning curves and generated jokes saved.")
