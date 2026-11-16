import numpy as np
import torch
from torch import nn

from local_code.base_class.method import method


class Method_RNN_Classification(method, nn.Module):
    data = None
    max_epoch = 5
    learning_rate = 0.001
    batch_size = 64

    
    def __init__(self, mName, mDescription, vocab_size,
                embedding_dim=100, hidden_dim=128,
                rnn_type="RNN"):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.rnn_type = rnn_type.upper()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )

        if self.rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True
            )

        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True
            )

        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True
            )

        else:
            raise ValueError("rnn_type must be RNN, LSTM, or GRU")

        self.fc = nn.Linear(hidden_dim, 2)

        self.loss_history = []
        self.train_acc_history = []


    
    def forward(self, x):
        # Count actual words, excluding padding
        lengths = (x != 0).sum(dim=1).clamp(min=1)

        # Convert word IDs into embeddings
        x = self.embedding(x)

        # Process the sequence with the RNN
        output, _ = self.rnn(x)

        # Select the hidden state at the last real word
        batch_indices = torch.arange(
            x.size(0), device=x.device
        )

        last_hidden = output[
            batch_indices, lengths - 1
        ]

        # Predict positive or negative sentiment
        return self.fc(last_hidden)


    def fit(self, X, y):
        nn.Module.train(self, True)
        X = torch.as_tensor(X, dtype=torch.long)
        y = torch.as_tensor(y, dtype=torch.long)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate
        )
        loss_function = nn.CrossEntropyLoss()

        self.loss_history = []
        self.train_acc_history = []

        for epoch in range(self.max_epoch):
            indices = torch.randperm(len(X))
            total_loss = 0.0
            correct = 0

            for start in range(0, len(X), self.batch_size):
                batch_indices = indices[start:start + self.batch_size]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                logits = self.forward(X_batch)
                loss = loss_function(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(X_batch)
                correct += (
                    logits.argmax(dim=1) == y_batch
                ).sum().item()

            avg_loss = total_loss / len(X)
            train_acc = correct / len(X)

            self.loss_history.append(avg_loss)
            self.train_acc_history.append(train_acc)

            print(
                f"Epoch {epoch + 1}/{self.max_epoch}: "
                f"Loss={avg_loss:.4f}, "
                f"Train Accuracy={train_acc:.4f}"
            )

    def test(self, X):
        nn.Module.train(self, False)
        X = torch.as_tensor(X, dtype=torch.long)
        predictions = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                X_batch = X[start:start + self.batch_size]
                logits = self.forward(X_batch)
                predictions.append(logits.argmax(dim=1))

        return torch.cat(predictions).numpy()

    def run(self):
        print("method running...")
        print("--start training...")

        self.fit(
            self.data["train"]["X"],
            self.data["train"]["y"]
        )

        print("--start testing...")

        pred_y = self.test(self.data["test"]["X"])

        return {
            "pred_y": pred_y,
            "true_y": self.data["test"]["y"],
            "train_loss": self.loss_history,
            "train_acc": self.train_acc_history
        }
