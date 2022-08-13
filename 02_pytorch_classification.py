# Classification
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/02_pytorch_classification.ipynb
# https://youtu.be/Z_ikDlimN6A?t=30690

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


def train_model(model, epochs, X_train, y_train, loss_fn, opt):
    epoch_counts = []
    train_loss_values = []
    test_loss_values = []
    for epoch in range(epochs):
        model.train() # sets up the parameters the require gradients
        # 1. Forward pass
        y_pred = model(X_train)
        # 2. Loss
        # print(torch.squeeze(y_pred).shape)
        # print(y_train.shape)
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



#loss_fn = nn.L1Loss() # MAE mean avg error
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model_0.parameters(), lr=0.01)

with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
y_logits

# so I thought he said something about the sigmoid being included but I guess not
torch.round(torch.sigmoid(y_logits))

# https://youtu.be/Z_ikDlimN6A?t=37338

(epochs, train_losses, test_losses) = train_model(model_0, 100, X_train, y_train, loss_fn, opt)



def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc













# Function inspired from Convolutional Neural Networks for Visual Recognition
# [cs231n](http://cs231n.stanford.edu/) 
# And
# https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb

def plot_decision_boundary(model, X, y):
    """
    Plots the decision boundary created by a model predicting on X.
    """
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
    y_min, y_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max),
                         np.linspace(y_min, y_max))
    
    # Create X values (we're going to make predictions on these)
    x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together

    # Make predictions
    y_pred = model.predict(x_in)

    # Check for multi-class
    if len(y_pred[0]) > 1:
        print("Doing multiclass classification")
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("Doing binary classification")
        y_pred = np.round(y_pred).reshape(xx.shape)
    
    # plot the decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

