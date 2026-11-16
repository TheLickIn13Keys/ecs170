from pathlib import Path
import sys
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_4_code.Dataset_Loader_Classification import (
    Dataset_Loader_Classification
)
from local_code.stage_4_code.Dataset_Loader_Generation import (
    Dataset_Loader_Generation
)
from local_code.stage_4_code.Method_RNN_Classification import (
    Method_RNN_Classification
)
from local_code.stage_4_code.Method_RNN_Generation import (
    Method_RNN_Generation
)


def set_seed(seed=2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


result_folder = PROJECT_ROOT / "result/stage_4_result"
result_folder.mkdir(parents=True, exist_ok=True)


# =====================================================
# Ablation 1: IMDb classification, hidden size 128 -> 64
# =====================================================

print("\n========== Classification Ablation ==========")

loader = Dataset_Loader_Classification("IMDb", "")
loader.dataset_source_folder_path = str(
    PROJECT_ROOT / "data/stage_4_data/text_classification"
)
classification_data = loader.load()

set_seed()

classification_model = Method_RNN_Classification(
    "GRU_Classification_Hidden64",
    "GRU classification with 64 hidden units",
    vocab_size=classification_data["vocab_size"],
    embedding_dim=100,
    hidden_dim=64,
    rnn_type="GRU"
)

classification_model.data = classification_data
classification_model.max_epoch = 5
classification_model.learning_rate = 0.001
classification_model.batch_size = 64

classification_result = classification_model.run()

classification_accuracy = accuracy_score(
    classification_result["true_y"],
    classification_result["pred_y"]
)

classification_f1 = f1_score(
    classification_result["true_y"],
    classification_result["pred_y"],
    average="macro",
    zero_division=0
)

print("\n=== Classification Ablation Results ===")
print(f"Test Accuracy: {classification_accuracy * 100:.2f}%")
print(f"Macro F1:      {classification_f1 * 100:.2f}%")


# =====================================================
# Ablation 2: Joke generation, embedding size 100 -> 50
# =====================================================

print("\n========== Generation Ablation ==========")

loader = Dataset_Loader_Generation("Jokes", "")
loader.dataset_source_folder_path = str(
    PROJECT_ROOT / "data/stage_4_data/text_generation/data"
)
generation_data = loader.load()

set_seed()

generation_model = Method_RNN_Generation(
    "GRU_Generation_Embedding50",
    "GRU generation with 50-dimensional embeddings",
    vocab_size=generation_data["vocab_size"],
    embedding_dim=50,
    hidden_dim=128,
    rnn_type="GRU"
)

generation_model.data = generation_data
generation_model.max_epoch = 20
generation_model.learning_rate = 0.001
generation_model.batch_size = 64

generation_result = generation_model.run()

print("\n=== Generation Ablation Results ===")
print(
    f"Final Training Loss: "
    f"{generation_result['train_loss'][-1]:.4f}"
)
print(
    f"Next-word Training Accuracy: "
    f"{generation_result['train_acc'][-1] * 100:.2f}%"
)

prompts = [
    "what did the",
    "why did the",
    "how many hipsters"
]

output_path = result_folder / "GRU_Ablation_Generation_examples.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for prompt in prompts:
        generated = generation_model.generate(
            prompt,
            generation_data["word_to_id"],
            generation_data["id_to_word"],
            sequence_length=generation_data["sequence_length"],
            max_new_tokens=40
        )

        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")

        f.write(f"Prompt: {prompt}\n")
        f.write(f"Generated: {generated}\n\n")

print("\nAblation experiments completed!")
