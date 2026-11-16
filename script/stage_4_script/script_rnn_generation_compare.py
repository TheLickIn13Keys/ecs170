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

# Load the dataset once
loader = Dataset_Loader_Generation("Jokes", "")
loader.dataset_source_folder_path = str(
    PROJECT_ROOT / "data/stage_4_data/text_generation/data"
)
data = loader.load()

result_folder = PROJECT_ROOT / "result/stage_4_result"
result_folder.mkdir(parents=True, exist_ok=True)

prompts = [
    "what did the",
    "why did the",
    "how many hipsters"
]

for rnn_type in ["LSTM", "GRU"]:
    print(f"\n========== {rnn_type} Generation ==========")

    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)

    model = Method_RNN_Generation(
        f"{rnn_type}_Generation",
        f"Joke generation using {rnn_type}",
        vocab_size=data["vocab_size"],
        embedding_dim=100,
        hidden_dim=128,
        rnn_type=rnn_type
    )

    model.data = data
    model.max_epoch = 20
    model.learning_rate = 0.001
    model.batch_size = 64

    result = model.run()

    # Training loss
    plt.figure()
    plt.plot(
        range(1, model.max_epoch + 1),
        result["train_loss"],
        marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"{rnn_type} Joke Generation Loss")
    plt.grid(True)
    plt.savefig(
        result_folder / f"{rnn_type}_Generation_loss.png"
    )
    plt.close()

    # Next-word training accuracy
    plt.figure()
    plt.plot(
        range(1, model.max_epoch + 1),
        np.array(result["train_acc"]) * 100,
        marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Next-word Accuracy (%)")
    plt.title(f"{rnn_type} Joke Generation Accuracy")
    plt.grid(True)
    plt.savefig(
        result_folder / f"{rnn_type}_Generation_accuracy.png"
    )
    plt.close()

    # Generate and save examples
    output_path = (
        result_folder / f"{rnn_type}_Generation_examples.txt"
    )

    with open(output_path, "w", encoding="utf-8") as f:
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

            f.write(f"Starting words: {prompt}\n")
            f.write(f"Generated joke: {generated}\n\n")

print("\nFinished LSTM and GRU generation experiments!")
