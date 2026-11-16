import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset


class Dataset_Loader_Generation(dataset):

    def load(self):
        print("Loading joke generation data...")

        file_path = Path(self.dataset_source_folder_path)

        jokes = []

        # Read the CSV file
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                joke = row["Joke"].strip()

                if joke:
                    jokes.append(joke)

        print(f"Total jokes: {len(jokes)}")

        # Tokenize while preserving punctuation
        def tokenize(text):
            text = text.lower()
            return re.findall(r"\w+|[^\w\s]", text)

        tokenized_jokes = [tokenize(joke) for joke in jokes]

        # Build vocabulary
        word_counts = Counter(
            word
            for joke in tokenized_jokes
            for word in joke
        )

        word_to_id = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<EOS>": 2
        }

        for word, _ in word_counts.most_common():
            word_to_id[word] = len(word_to_id)

        id_to_word = {
            idx: word
            for word, idx in word_to_id.items()
        }

        # Create input-target sequences for next-word prediction
        sequence_length = 10
        X = []
        y = []

        for joke in tokenized_jokes:
            tokens = [
                word_to_id.get(word, word_to_id["<UNK>"])
                for word in joke
            ]

            tokens.append(word_to_id["<EOS>"])

            # Use up to 10 preceding tokens to predict the next
            for i in range(1, len(tokens)):
                context = tokens[max(0, i - sequence_length):i]

                padded = (
                    [word_to_id["<PAD>"]] *
                    (sequence_length - len(context))
                    + context
                )

                X.append(padded)
                y.append(tokens[i])

        X = np.array(X, dtype=np.int64)
        y = np.array(y, dtype=np.int64)

        print(f"Vocabulary size: {len(word_to_id)}")
        print(f"Training sequences: {len(X)}")
        print(f"Input shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        return {
            "train": {"X": X, "y": y},
            "vocab_size": len(word_to_id),
            "word_to_id": word_to_id,
            "id_to_word": id_to_word,
            "jokes": jokes,
            "sequence_length": sequence_length
        }
