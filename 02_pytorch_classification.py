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

