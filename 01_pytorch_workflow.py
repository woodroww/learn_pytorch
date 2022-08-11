## Daniel's Notebook
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/01_pytorch_workflow.ipynb

# ------------------------------------------------------------------------------
# PyTorch Workflow
# ------------------------------------------------------------------------------
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

from pathlib import Path
MODEL_DIR = "/Users/matt/prog/torch_daniel/models"

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

# ------------------------------------------------------------------------------
# Visualize, Visualize, Visualize
# ------------------------------------------------------------------------------


def plot_predictions(
    train_data,
    train_labels,
    test_data,
    test_labels,
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


# ------------------------------------------------------------------------------
# Build a model
# ------------------------------------------------------------------------------
# Linear regression
# Y = a + bX
# In PyTorch we create a python subclass
# https://realpython.com/python3-object-oriented-programming/
# backpropagation
# https://youtu.be/IHZwWFHWa-w

from torch import nn

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



# ------------------------------------------------------------------------------
# PyTorch model building essentails
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
## Making predictions
# ------------------------------------------------------------------------------
# ??? I thought I'd try using just the forward method
# but maybe this is updating the weights too and this is not what we want
# when we want to infer
# maybe_preds = model_0.forward(X_test)
# y_test

# use inference mode no_grad() also exists but who knows
with torch.inference_mode():
    y_preds = model_0(X_test)
print(y_preds)
print(y_preds - y_test)
plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=y_preds)

## So now we will actually train the model
# We need a loss function
# https://pytorch.org/docs/stable/nn.html#loss-functions
# and an optimizer
# https://pytorch.org/docs/stable/optim.html?highlight=optimizer#torch.optim.Optimizer

# MAE_loss = torch.mean(torch.abs(y_pred-y_test))
# or
# MAE_loss = torch.nn.L1Loss

loss_fn = nn.L1Loss() # MAE mean avg error
opt = torch.optim.SGD(model_0.parameters(), lr=0.01)

# ------------------------------------------------------------------------------
## Training loop
# ------------------------------------------------------------------------------
# 0. Loop
# 1. Forward pass
# 2. Calculate the loss
# 3. Optimizer zero grad
# 4. Loss backward - back propagation
# 5. Gradient descent, Optimizer step - optimizer adjusts the weights (the model's parameters)

# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
# idk did Daniel make a mess idk
# pytorch website
#       optimizer.zero_grad()
#       # Make predictions for this batch
#       outputs = model(inputs)
#       # Compute the loss and its gradients
#       loss = loss_fn(outputs, labels)
#       loss.backward()
#       # Adjust learning weights
#       optimizer.step()

def train_model(model, epochs, X_train, y_train, loss_fn, opt):
    epoch_counts = []
    train_loss_values = []
    test_loss_values = []
    for epoch in range(epochs):
        model.train() # sets up the parameters the require gradients
        # 1. Forward pass
        y_pred = model(X_train)
        # 2. Loss
        loss = loss_fn(y_pred, y_train)
        # 3. Optimizer
        opt.zero_grad() # zero out optimizer changes
        # 4. Backpropagation
        loss.backward()
        # 5. Gradient descent
        opt.step() # accumulate changes here
        #print(model.state_dict())
        model.eval() # turns off gradient tracking
        with torch.inference_mode():
            # forward pass
            test_pred = model(X_test)
            # calculate the loss
            test_loss = loss_fn(test_pred, y_test)
        if epoch % 10 == 0:
            epoch_counts.append(epoch)
            train_loss_values.append(loss)
            test_loss_values.append(test_loss)
            print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
    return (epoch_counts, train_loss_values, test_loss_values)

# ------------------------------------------------------------------------------
## Train a model
# ------------------------------------------------------------------------------
torch.manual_seed(42)
model_0 = LinearRegressionModel()
with torch.inference_mode():
    y_preds = model_0(X_test)
# plot predictions before any training
plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=y_preds)
# loss and optimizer
loss_fn = nn.L1Loss() # MAE mean avg error
opt = torch.optim.SGD(model_0.parameters(), lr=0.01)
# train loop
(epochs, train_losses, test_losses) = train_model(model_0, 160, X_train, y_train, loss_fn, opt)
# make predictions after training
with torch.inference_mode():
    y_preds_new = model_0(X_test)
plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=y_preds)


# ------------------------------------------------------------------------------
## evaluation - turns off dropout and batch norm and who knows what else
# ------------------------------------------------------------------------------
# eval()
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=eval#torch.nn.Module.eval
# inference_mode()
# https://pytorch.org/docs/stable/generated/torch.inference_mode.html?highlight=inference_mode#torch.inference_mode

plt.plot(epochs, np.array(torch.tensor(train_losses).numpy()), label="Train Loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.title("Train and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()

# https://youtu.be/Z_ikDlimN6A?t=26130

# ------------------------------------------------------------------------------
# Saving a model
# ------------------------------------------------------------------------------
# https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

# 1. torch.save() - python pickle
# 2. torch.load() - load
# 3. torch.nn.load_state_dict() - load model's saved state dictionary

model_0.state_dict()

model_path = Path(MODEL_DIR)

model_name = "01_pytorch_workflow_model_0.pth"
model_save_path = model_path / model_name

# save the state dict
torch.save(model_0.state_dict(), model_save_path)
print(f"Saved model to: {model_save_path}")

# ------------------------------------------------------------------------------
# Load a model
# ------------------------------------------------------------------------------
# We only saved the state dict so we must:
# - create a new model
# - load the state dict

loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(model_save_path))

# try some predictions
loaded_model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)
    load_preds = loaded_model_0(X_test)

y_preds == load_preds


# ------------------------------------------------------------------------------
# Putting it together
# ------------------------------------------------------------------------------
# https://youtu.be/Z_ikDlimN6A?t=27481

import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# for device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.device(device)

# make linear regression input data (features)
# Y = a + bX
weight = 0.7
bias = 0.1
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# split the data
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, shuffle=True)

# maybe one day we will have a gpu
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# visualize
plot_predictions(X_train, y_train, X_test, y_test)

# create model
class LinearRegressionV2(nn.Module):
    def __init__(self):
        super().__init__()
        # this time use torch linear layer
        # 1, 1 are the shapes of the features and labels
        # nn.Linear will do the weights and bias initialization we did before
        # and use the linear regression formula
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# reproducability
torch.manual_seed(42)
model_1 = LinearRegressionV2()
model_1.state_dict()

# for device agnostic code
model_1.to(device)

# check and see if untrained model has reasonable output
model_1.eval()
with torch.inference_mode():
    y_preds = model_1(X_test)
plot_predictions(X_train, y_train, X_test, y_test, y_preds)

# setup loss function and optimizer
# could these be in the LinearRegressionV2 model ???
loss_fn = nn.L1Loss() # MAE mean avg error
opt = torch.optim.SGD(model_1.parameters(), lr=0.01)

# train using the training loop
(epochs, train_losses, test_losses) = train_model(model_1, 100, X_train, y_train, loss_fn, opt)

# evaluate performance
model_1.eval()
with torch.inference_mode():
    y_preds = model_1(X_test)
plot_predictions(X_train, y_train, X_test, y_test, y_preds)

# Y = a + bX
# weight = 0.7
# bias = 0.1
model_1.state_dict()

# plot loss curves
fig = plt.figure()
plt.plot(epochs, np.array(torch.tensor(train_losses).numpy()), label="Train Loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.title("Train and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()

# https://youtu.be/Z_ikDlimN6A?t=29584

# save the model
model_name = "01_pytorch_workflow_model_1.pth"
model_save_path = Path(MODEL_DIR) / model_name
torch.save(model_1.state_dict(), model_save_path)
print(f"Saved model to: {model_save_path}")

# load the model
loaded_model_1 = LinearRegressionV2()
loaded_model_1.load_state_dict(torch.load(model_save_path))

# agnostic device
loaded_model_1.to(device)

# try some predictions
loaded_model_1.eval()
with torch.inference_mode():
    load_preds = loaded_model_1(X_test)

# see if we have the same predictions
y_preds == load_preds

# https://youtu.be/Z_ikDlimN6A?t=30487
# Extra curriculum
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/01_pytorch_workflow.ipynb



