
import re
from collections import Counter
from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset


class Dataset_Loader_Classification(dataset):

    def load(self):
        print("Loading text classification data...")

        folder = Path(self.dataset_source_folder_path)

        def read_reviews(split):
            texts = []
            labels = []

            for label_name, label in [("neg", 0), ("pos", 1)]:
                path = folder / split / label_name

                for file in sorted(path.glob("*.txt")):
                    text = file.read_text(
                        encoding="utf-8", errors="replace"
                    )
                    texts.append(text)
                    labels.append(label)

            return texts, np.array(labels)

        train_texts, train_y = read_reviews("train")
        test_texts, test_y = read_reviews("test")

        # Convert text into lowercase word tokens
        def tokenize(text):
            text = text.lower()
            text = re.sub(r"<br\s*/?>", " ", text)
            return re.findall(r"\b\w+\b", text)

        train_tokens = [tokenize(text) for text in train_texts]
        test_tokens = [tokenize(text) for text in test_texts]

        # Build vocabulary from training data only
        word_counts = Counter(
            word for tokens in train_tokens for word in tokens
        )

        vocab_size = 20000
        word_to_id = {"<PAD>": 0, "<UNK>": 1}

        for word, _ in word_counts.most_common(vocab_size - 2):
            word_to_id[word] = len(word_to_id)

        # Convert each review into a fixed-length sequence
        max_length = 200

        def encode(tokens):
            ids = [
                word_to_id.get(word, word_to_id["<UNK>"])
                for word in tokens[:max_length]
            ]

            ids += [word_to_id["<PAD>"]] * (max_length - len(ids))
            return ids

        train_X = np.array(
            [encode(tokens) for tokens in train_tokens],
            dtype=np.int64
        )
        test_X = np.array(
            [encode(tokens) for tokens in test_tokens],
            dtype=np.int64
        )

        print(f"Training reviews: {len(train_X)}")
        print(f"Testing reviews: {len(test_X)}")
        print(f"Vocabulary size: {len(word_to_id)}")

        return {
            "train": {"X": train_X, "y": train_y},
            "test": {"X": test_X, "y": test_y},
            "vocab_size": len(word_to_id),
            "word_to_id": word_to_id
        }
