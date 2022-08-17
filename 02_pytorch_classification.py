# Classification
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/02_pytorch_classification.ipynb
# https://youtu.be/Z_ikDlimN6A?t=30690

from numpy.core.fromnumeric import squeeze
import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from helper_functions import plot_predictions

MODEL_DIR = "/Users/matt/prog/torch_daniel/models"
plt.ion()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.device(device)

# Architectured of a classification problem
# https://learnpytorch.io/02_pytorch_classification/#0-architecture-of-a-classification-neural-network

# Create some sample data

n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# plot the data
# y is the color of the dot
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()

# look at the data
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
circles

# https://youtu.be/Z_ikDlimN6A?t=33107

X.shape
y.shape

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

# split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=True, random_state=42
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

torch.manual_seed(42)

# So we want to classify the blue dots as blue and the red dots as red


class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))


model_0 = CircleModelV0().to(device)
model_0
model_0.state_dict()

model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)
torch.round(y_preds)

## yikes

# loss_fn = nn.L1Loss() # MAE mean avg error
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model_0.parameters(), lr=0.01)

with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
y_logits

# so I thought he said something about the sigmoid being included but I guess not
y_pred_probs = torch.sigmoid(y_logits)

y_preds = torch.round(y_pred_probs)

# https://youtu.be/Z_ikDlimN6A?t=37338

y_preds[:5]

torch.manual_seed(42)
torch.cuda.manual_seed(42)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# argument order based off how sci kit does it
# pytorch is reversed
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def train_model(model, epochs, X_train, y_train, loss_fn, opt):
    epoch_counts = []
    train_loss_values = []
    test_loss_values = []
    for epoch in range(epochs):
        model.train()  # sets up the parameters the require gradients
        # 1. Forward pass
        y_logits = model(X_train).squeeze()
        y_preds = torch.round(torch.sigmoid(y_logits))
        # 2. Loss
        # from docs BCEWithLogitsLoss combines a sigmoid layer and BCELoss in a single class
        # expects raw logits
        loss = loss_fn(y_logits, y_train)
        # BCELoss would be
        # loss = loss_fn(y_preds, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_preds)
        # 3. Optimizer
        opt.zero_grad()  # zero out optimizer changes
        # 4. Backpropagation
        loss.backward()
        # 5. Gradient descent
        opt.step()  # accumulate changes here
        # Testing
        model.eval()  # turns off gradient tracking
        with torch.inference_mode():
            # forward pass
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            # calculate the loss
            test_loss = loss_fn(test_logits, y_test)
            test_accuracy = accuracy_fn(y_true=y_test, y_pred=test_pred)
        if epoch % 10 == 0:
            epoch_counts.append(epoch)
            train_loss_values.append(loss)
            test_loss_values.append(test_loss)
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.5f} | Test loss: {test_loss:.5f} | Test acc: {test_accuracy:.5f}"
            )
    return (epoch_counts, train_loss_values, test_loss_values)


(epochs, train_losses, test_losses) = train_model(
    model_0, 100, X_train, y_train, loss_fn, opt
)

# So not really good

from helper_functions import plot_decision_boundary


def decision_plots(model, X_train, y_train, X_test, y_test):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, X_test, y_test)


decision_plots(model_0, X_train, y_train, X_test, y_test)

# https://youtu.be/Z_ikDlimN6A?t=40301

# Here we shall change one variable here to improve our model
# Why Daniel talk about this then he changes two things
# I think I only increase the features here
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=20)
        self.layer_2 = nn.Linear(in_features=20, out_features=1)

    def forward(self, x):
        z = self.layer_1(x)
        z = self.layer_2(z)
        return z


model_1 = CircleModelV1().to(device)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model_1.parameters(), lr=0.01)
(epochs, train_losses, test_losses) = train_model(
    model_1, 100, X_train, y_train, loss_fn, opt
)
decision_plots(model_1, X_train, y_train, X_test, y_test)

# https://youtu.be/Z_ikDlimN6A?t=40979
# Begin a diversion back into regression
# ------------------------------------------------------------------------------
# So we know the problem is we don't have any non-linearity
# Daniel wants to trouble shoot this problem step by step
# One way to troubleshoot a larger problem is to test out a smaller problem
# Can this model fit a linear regression? Let's find out.

weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = bias + weight * X_regression

(
    X_regression_train,
    X_regression_test,
    y_regression_train,
    y_regression_test,
) = train_test_split(
    X_regression, y_regression, test_size=0.20, shuffle=False, random_state=42
)

X_regression_train = X_regression_train.to(device)
y_regression_train = y_regression_train.to(device)
X_regression_test = X_regression_test.to(device)
y_regression_test = y_regression_test.to(device)
len(X_regression_train)
len(y_regression_train)
len(X_regression_test)
len( y_regression_test)

plot_predictions(
    train_data=X_regression_train,
    train_labels=y_regression_train,
    test_data=X_regression_test,
    test_labels=y_regression_test,
)

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=1, out_features=20)
        self.layer_2 = nn.Linear(in_features=20, out_features=1)
    def forward(self, x):
        z = self.layer_1(x)
        z = self.layer_2(z)
        return z

# train model
regression_model = RegressionModel().to(device)
loss_fn = nn.L1Loss()
opt = torch.optim.SGD(regression_model.parameters(), lr=0.01)
for epoch in range(1000):
    y_pred = regression_model(X_regression_train)
    loss = loss_fn(y_pred, y_regression_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    regression_model.eval()
    with torch.inference_mode():
        test_pred = regression_model(X_regression_test)
        test_loss = loss_fn(test_pred, y_regression_test)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss}")

regression_model.eval()
with torch.inference_mode():
    y_preds = regression_model(X_regression_test)

plot_predictions(
    train_data=X_regression_train,
    train_labels=y_regression_train,
    test_data=X_regression_test,
    test_labels=y_regression_test,
    predictions=y_preds
)

# https://youtu.be/Z_ikDlimN6A?t=42096
# end diversion
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Non-linear activations
# ------------------------------------------------------------------------------
# So we know the problem is we don't have any non-linearity
## add a relu layer

# try to resart everything here for ease of use

import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary
import matplotlib.pyplot as plt

plt.ion()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.device(device)

n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=False, random_state=42
)

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
        # could do this here and remove it from the train_model function
        # self.sig = nn.Sigmoid()
    def forward(self, x):
        z = self.layer_1(x)
        z = self.relu(z)
        z = self.layer_2(z)
        z = self.relu(z)
        z = self.layer_3(z)
        return z

model_3 = CircleModelV2().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(), lr=0.01)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

for epoch in range(1000):
    # 1. Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
    
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_3.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model_3(X_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
      # 2. Calcuate loss and accuracy
      test_loss = loss_fn(test_logits, y_test)
      test_acc = accuracy_fn(y_true=y_test,
                             y_pred=test_pred)

    # Print out what's happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

decision_plots(model_3, X_train, y_train, X_test, y_test)
plt.show()

# https://youtu.be/Z_ikDlimN6A?t=44806


# TODO:
# create a class to hold the results from train model like tensorflow history
# create a function for the above where we create the loss fn and the optimizer

