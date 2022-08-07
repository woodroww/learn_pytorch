import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Daniel's Notebook
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/01_pytorch_workflow.ipynb

# PyTorch Workflow
# 1. Get data ready (data cleaning and transformation, turn into tensors)
# 2. Build or pick a model
#   2.1 Pick a loss function and optimizer
#   2.2 Build a training loop
# 3. Fit the model and make a prediction
# 4. Evaluate the model
# 5. Improve through experimentation
# 6. Save and reload your trained model

## Create some known data using the linear regression formula
# A linear regression line has an equation of the form
# Y = a + bX
# where X is the explanatory variable and Y is the dependent variable. The slope
# of the line is b, and a is the intercept (the value of y when X = 0)

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# known parameters
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = bias + weight * X

X[:10], y[:10]

# premise of machine learning here is to predict y from X
# y from above is the ground truth

# https://youtu.be/Z_ikDlimN6A?t=16566

# Split data into training and test sets

## manually
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)

# Visualize, Visualize, Visualize


def plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=None,
):
    """
    Plots training data, test data and compares predictions
    """
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")

    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


plt.ion()
plot_predictions(
    train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test
)


# Build a model
# Linear regression
# Y = a + bX
# In PyTorch we create a python subclass
# https://realpython.com/python3-object-oriented-programming/
# backpropagation
# https://youtu.be/IHZwWFHWa-w

import torch.nn as nn


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.parameter.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float)
        )
        self.bias = nn.parameter.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias



# PyTorch model building essentails
# torch.nn
# Contains all the building blocks for computational graphs
# torch.nn.Module
# The base class for all neural network modules
# torch.optim
# Contains various optimization algorithms
# torch.utils.data.Dataset
# Represents a map between label and feature pairs of your data
# torch.utils.data.DataLoader
# Creates a python iterable over a torch dataset

# https://pytorch.org/tutorials/beginner/ptcheat.html

# https://youtu.be/Z_ikDlimN6A?t=19164

## Check the contents of our model
# set seed
torch.manual_seed(42)
# create model of our class
model_0 = LinearRegressionModel()
# check the parameters
list(model_0.parameters())

model_0.weights
model_0.bias

model_0.state_dict()













