'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None

    # 784 features
    input_dim = 784
    output_dim = 10

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, hidden_dims=(256, 128), activation='relu',
                 loss='cross_entropy', optimizer='adam', learning_rate=1e-3,
                 max_epoch=20, batch_size=128):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.loss_name = loss
        self.optimizer_name = optimizer
        # it defines the learning rate for gradient descent based optimizer for model learning
        self.learning_rate = learning_rate
        # Number of training epochs
        self.max_epoch = max_epoch
        self.batch_size = batch_size

        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        activations = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        layers = []
        in_dim = self.input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activations[activation]())
            in_dim = h_dim
        # 10 output as per "the first element denotes the label with value from {0, 1, 2, …, 9"
        layers.append(nn.Linear(in_dim, self.output_dim))
        # CrossEntropyLoss takes raw logits, so no output softmax is needed.
        self.layers = nn.Sequential(*layers)

        self.loss_history = []
        self.train_acc_history = []

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        # output layer result
        # n instances along dimension 0; 10 classes along dimension 1
        return self.layers(x)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def get_loss_function(self):
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        if self.loss_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        if self.loss_name == 'multi_margin':
            return nn.MultiMarginLoss()
        if self.loss_name == 'mse':
            mse = nn.MSELoss()
            return lambda y_pred, y_true: mse(torch.softmax(y_pred, dim=1),
                                              nn.functional.one_hot(y_true, self.output_dim).float())
        raise ValueError('unknown loss: ' + self.loss_name)

    def get_optimizer(self):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        if self.optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        if self.optimizer_name == 'adagrad':
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        raise ValueError('unknown optimizer: ' + self.optimizer_name)

    def train(self, X, y):
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # Convert and normalize once; the same tensors are used each epoch.
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32) / 255.0
        y_true = torch.LongTensor(np.array(y))
        n = len(y_true)
        batch_size = self.batch_size or n

        self.loss_history = []
        self.train_acc_history = []
        # it will be an iterative gradient updating process
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            permutation = torch.randperm(n)
            epoch_loss = 0.0
            epoch_pred = torch.empty(n, dtype=torch.long)
            for start in range(0, n, batch_size):
                idx = permutation[start:start + batch_size]
                y_pred = self.forward(X_tensor[idx])
                # calculate the training loss
                train_loss = loss_function(y_pred, y_true[idx])

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()

                epoch_loss += train_loss.item() * len(idx)
                epoch_pred[idx] = y_pred.detach().max(1)[1]

            # Log training accuracy and loss each epoch
            accuracy_evaluator.data = {'true_y': y_true, 'pred_y': epoch_pred}
            self.loss_history.append(epoch_loss / n)
            self.train_acc_history.append(accuracy_evaluator.evaluate())
            print('Epoch:', epoch, 'Accuracy:', self.train_acc_history[-1], 'Loss:', self.loss_history[-1])

    def test(self, X):
        # do the testing, and result the result

        # Scale integer pixels to [0, 1]
        # Testing does not need gradient tracking
        with torch.no_grad():
            X_tensor = torch.FloatTensor(np.array(X)) / 255.0
            y_pred = self.forward(X_tensor)
        # Choose the class with the largest logit
        # Argmax is the same with or without softmax
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'],
                'train_loss': self.loss_history, 'train_acc': self.train_acc_history}
