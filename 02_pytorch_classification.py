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
        X, y, test_size=0.20, shuffle=True, random_state=42)

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

#loss_fn = nn.L1Loss() # MAE mean avg error
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
        model.train() # sets up the parameters the require gradients
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
        opt.zero_grad() # zero out optimizer changes
        # 4. Backpropagation
        loss.backward()
        # 5. Gradient descent
        opt.step() # accumulate changes here
        # Testing
        model.eval() # turns off gradient tracking
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
            print(f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.5f} | Test loss: {test_loss:.5f} | Test acc: {test_accuracy:.5f}")
    return (epoch_counts, train_loss_values, test_loss_values)

(epochs, train_losses, test_losses) = train_model(model_0, 100, X_train, y_train, loss_fn, opt)

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
(epochs, train_losses, test_losses) = train_model(model_1, 100, X_train, y_train, loss_fn, opt)
decision_plots(model_1, X_train, y_train, X_test, y_test)

# https://youtu.be/Z_ikDlimN6A?t=40979

## add a relu layer
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=20)
        self.layer_2 = nn.ReLU()
        self.layer_3 = nn.Linear(in_features=20, out_features=1)
    def forward(self, x):
        z = self.layer_1(x)
        z = self.layer_2(z)
        z = self.layer_3(z)
        return z
        
model_2 = CircleModelV2().to(device)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model_2.parameters(), lr=0.01) 
(epochs, train_losses, test_losses) = train_model(model_2, 100, X_train, y_train, loss_fn, opt)
decision_plots(model_2, X_train, y_train, X_test, y_test)

# create a class to hold the results from train model like tensorflow history
# create a function for the above where we create the loss fn and the optimizer

class CircleModelV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=20)
        self.layer_2 = nn.ReLU()
        self.layer_3 = nn.Linear(in_features=20, out_features=40)
        self.layer_4 = nn.ReLU()
    def forward(self, x):
        z = self.layer_1(x)
        z = self.layer_2(z)
        z = self.layer_3(z)
        z = self.layer_4(z)
        z = self.layer_5(z)
        z = self.layer_6(z)
        return z
        
model_3 = CircleModelV3().to(device)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model_3.parameters(), lr=0.01) 
(epochs, train_losses, test_losses) = train_model(model_3, 100, X_train, y_train, loss_fn, opt)
decision_plots(model_3, X_train, y_train, X_test, y_test)




