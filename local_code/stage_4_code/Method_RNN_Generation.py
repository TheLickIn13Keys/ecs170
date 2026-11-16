import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from local_code.base_class.method import method


class Method_RNN_Generation(method, nn.Module):
    data = None
    max_epoch = 20
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
                embedding_dim, hidden_dim, batch_first=True
            )
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                embedding_dim, hidden_dim, batch_first=True
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                embedding_dim, hidden_dim, batch_first=True
            )
        else:
            raise ValueError("Choose RNN, LSTM, or GRU")

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.loss_history = []
        self.accuracy_history = []

    def forward(self, x):
        embedded = self.embedding(x)

        output, _ = self.rnn(embedded)

        # Our generation inputs are LEFT-padded, so the
        # final position is always the last actual token.
        last_hidden = output[:, -1, :]

        return self.fc(last_hidden)

    def fit(self, X, y):
        nn.Module.train(self, True)

        X = torch.as_tensor(X, dtype=torch.long)
        y = torch.as_tensor(y, dtype=torch.long)

        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=self.batch_size,
            shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate
        )
        loss_function = nn.CrossEntropyLoss()

        self.loss_history = []
        self.accuracy_history = []

        for epoch in range(self.max_epoch):
            total_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in loader:
                logits = self.forward(X_batch)
                loss = loss_function(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()

                # Help prevent exploding gradients
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                optimizer.step()

                total_loss += loss.item() * len(X_batch)
                correct += (
                    logits.argmax(dim=1) == y_batch
                ).sum().item()
                total += len(X_batch)

            avg_loss = total_loss / total
            accuracy = correct / total

            self.loss_history.append(avg_loss)
            self.accuracy_history.append(accuracy)

            print(
                f"Epoch {epoch + 1}/{self.max_epoch}: "
                f"Loss={avg_loss:.4f}, "
                f"Next-word Accuracy={accuracy:.4f}"
            )

    def generate(self, prompt, word_to_id, id_to_word,
                 sequence_length=10, max_new_tokens=40):

        nn.Module.train(self, False)

        # Match the loader's tokenization
        import re
        tokens = re.findall(r"\w+|[^\w\s]", prompt.lower())

        if not tokens:
            raise ValueError("Prompt must contain words")

        generated = tokens.copy()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = generated[-sequence_length:]

                ids = [
                    word_to_id.get(token, word_to_id["<UNK>"])
                    for token in context
                ]

                ids = [0] * (sequence_length - len(ids)) + ids

                x = torch.tensor([ids], dtype=torch.long)
                logits = self.forward(x)

                # Do not generate padding or unknown tokens
                logits[0, word_to_id["<PAD>"]] = float("-inf")
                logits[0, word_to_id["<UNK>"]] = float("-inf")

                next_id = logits.argmax(dim=1).item()
                next_word = id_to_word[next_id]

                if next_word == "<EOS>":
                    break

                generated.append(next_word)

        # Basic punctuation cleanup
        text = " ".join(generated)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        return text

    def run(self):
        print(f"Training {self.rnn_type} generator...")

        self.fit(
            self.data["train"]["X"],
            self.data["train"]["y"]
        )

        return {
            "train_loss": self.loss_history,
            "train_acc": self.accuracy_history
        }
