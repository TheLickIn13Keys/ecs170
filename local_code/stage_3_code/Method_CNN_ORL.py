
import numpy as np
import torch
from torch import nn

from local_code.base_class.method import method


class Method_CNN_ORL(method, nn.Module):
    data = None
    max_epoch = 30
    learning_rate = 1e-3
    batch_size = 32

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 28 * 23, 128),
            nn.ReLU(),
            nn.Linear(128, 40)
        )

        self.loss_history = []
        self.train_acc_history = []

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def prepare_images(self, X):
        X = np.asarray(X)

        # ORL: RGB channels are identical, so use one
        X = X[:, :, :, 0]

        X = torch.as_tensor(X, dtype=torch.float32)
        X = X / 255.0

        # (N, 112, 92) -> (N, 1, 112, 92)
        return X.unsqueeze(1)

    def train(self, X, y):
        X = self.prepare_images(X)

        # Convert labels 1-40 to 0-39
        y = torch.as_tensor(np.asarray(y) - 1, dtype=torch.long)

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

                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(X_batch)
                correct += (
                    y_pred.argmax(dim=1) == y_batch
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
        X = self.prepare_images(X)

        predictions = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                X_batch = X[start:start + self.batch_size]
                logits = self.forward(X_batch)

                # Convert predictions back to labels 1-40
                predictions.append(logits.argmax(dim=1) + 1)

        return torch.cat(predictions).numpy()

    def run(self):
        print('method running...')
        print('--start training...')

        self.train(
            self.data['train']['X'],
            self.data['train']['y']
        )

        print('--start testing...')

        pred_y = self.test(self.data['test']['X'])

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y'],
            'train_loss': self.loss_history,
            'train_acc': self.train_acc_history
        }
