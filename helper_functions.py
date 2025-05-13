# Function inspired from Convolutional Neural Networks for Visual Recognition
# http://cs231n.stanford.edu
# And
# https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
# and
# https://madewithml.com/courses/foundations/neural-networks/

# for PyTorch

import torch
import matplotlib.pyplot as plt
import numpy as np
# from torch import nn


# plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test)

def plot_predictions(
    train_data,
    train_labels,
    test_data,
    test_labels,
    predictions=None):
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


def plot_decision_boundary(
        model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """
    Plots the decision boundary created by a model predicting on X.
    Source - https://madewithml.com/courses/foundations/neural-networks
    """
    # device agnostic but here CPU works better with NumPy and Matplotlib
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                         np.linspace(y_min, y_max, 101))

    # Create X values (we're going to make predictions on these features)
    # x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
