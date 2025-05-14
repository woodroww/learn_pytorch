# Classification
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/02_pytorch_classification.ipynb

# beginning of classification
# https://youtu.be/Z_ikDlimN6A?t=30690
# beginning of circles code
# https://youtu.be/Z_ikDlimN6A?si=oai0PsMMgqZZCd4K&t=32299
# 2025 https://youtu.be/LyJtbe__2i0?si=JF0oWfGSM-xX-g4b&t=32999

import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary
from sklearn.datasets import make_blobs
from torchmetrics import Accuracy

# from numpy.core.fromnumeric import squeeze
# import numpy as np
# from pathlib import Path
# import sklearn

print(torch.__version__)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

plt.ion()
plt.style.use("/Users/matt/.config/matplotlib/stylelib/gruvbox.mplstyle")

MODEL_DIR = "/Users/matt/prog/ml/torch_daniel/models"

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
# 2025 https://youtu.be/LyJtbe__2i0?si=BcxUBQS3FM4ZIply&t=33676

X.shape, y.shape

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

# split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=True, random_state=42
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

torch.manual_seed(42)


# So we want to classify the blue dots as blue and the red dots as red
# 2025 https://youtu.be/LyJtbe__2i0?si=G4Tk1F07NHs1CDwG&t=34777
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
    untrained_logits = model_0(X_test.to(device))

untrained_logits.shape
untrained_logits[:5]

# https://medium.com/data-science/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model_0.parameters(), lr=0.01)

# convert the output of the model (the logits) into prediction probabilities
# by passing them to some kind of activation function
# sigmoid for binary classification
# softmax for multiclass classification
y_pred_probs = torch.sigmoid(untrained_logits)
y_pred_probs[:5]

# y_pred_probs >= 0.5, y=1 (class 1)
# y_pred_probs < 0.5, y=0 (class 0)
y_preds = torch.round(y_pred_probs)

y_preds[:5], y_test[:5]


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
        # from docs BCEWithLogitsLoss combines a sigmoid layer and BCELoss in a
        # single class
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


# From the metrics it looks like our model isn't learning anything
# So let's visualize
# https://madewithml.com/courses/foundations/neural-networks

def decision_plots(model, X_train, y_train, X_test, y_test):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, X_test, y_test)


decision_plots(model_0, X_train, y_train, X_test, y_test)
# so we can see that the circular data can't be seperated by a straight line


# so we try the same thing with one more layer, idk its Daniel
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        z = self.layer_1(x)
        z = self.layer_2(z)
        z = self.layer_3(z)
        return z


model_1 = CircleModelV1().to(device)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model_1.parameters(), lr=0.01)
(epochs, train_losses, test_losses) = train_model(
    model_1, 100, X_train, y_train, loss_fn, opt
)
decision_plots(model_1, X_train, y_train, X_test, y_test)


# ------------------------------------------------------------------------------
# Non-linear activations
# ------------------------------------------------------------------------------
# So we know the problem is we don't have any non-linearity
# 2025 https://youtu.be/LyJtbe__2i0?si=-W201m_rGoEANehv&t=44011
# https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
# add a relu layer

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

model_3.to(device)
for epoch in range(1000):
    model_3.train()
    # 1. Forward pass
    y_logits = model_3(X_train).squeeze()
    # logits -> prediction probabilities -> prediction labels
    y_pred = torch.round(torch.sigmoid(y_logits))

    # 2. Calculate loss and accuracy
    # BCEWithLogitsLoss calculates loss using logits
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Testing
    model_3.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_3(X_test).squeeze()
        # logits -> prediction probabilities -> prediction labels
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Calcuate loss and accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    # Print out what's happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

decision_plots(model_3, X_train, y_train, X_test, y_test)
plt.show()


# TODO:
# create a class to hold the results from train model like tensorflow history

# Activation plots

A = torch.arange(-10, 10, 1, dtype=torch.float32)
plt.plot(A)
plt.plot(torch.relu(A))


def relu(x):
    return torch.maximum(torch.tensor(0), x)


plt.plot(relu(A))

plt.plot(torch.sigmoid(A))


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


plt.plot(sigmoid(A))

# ------------------------------------------------------------------------------
# putting it all together
# ------------------------------------------------------------------------------
# multi-class classification

# create multi-class dataset

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.float)

plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)

X_blob = X_blob.to(device)
y_blob = y_blob.to(device)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
        X_blob,
        y_blob,
        test_size=0.2,
        random_state=RANDOM_SEED)


# 2025 https://youtu.be/LyJtbe__2i0?si=_lMHY-hwTEDpHubY&t=47163


# model
class BlobModel1(nn.Module):
    def __init__(self, input_features, ouput_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=ouput_features),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


model_4 = BlobModel1(input_features=2,
                     ouput_features=NUM_CLASSES).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_4.parameters(), lr=0.01)


model_4.eval()
with torch.inference_mode():
    test_logits = model_4(X_blob_test)
y_pred_probs = torch.softmax(test_logits, dim=1)

test_logits[:5]
y_pred_probs[:5]

# softmax sum results 1 for each sample
torch.sum(y_pred_probs[0])
# each value in the sample is a percentage of how likely that class is the answer
y_pred_probs[0]
y_class = 0
y_pred_probs[0, y_class]

most_likely_class = torch.argmax(y_pred_probs[0])
most_likely_class

# convert prediction probabilities to prediction labels
y_preds = torch.argmax(y_pred_probs, dim=1)


model_4.to(device)
for epoch in range(100):
    model_4.train()
    # 1. Forward pass
    y_logits = model_4(X_blob_train)
    # logits -> prediction probabilities -> prediction labels
    y_pred_probs = torch.softmax(y_logits, dim=1)
    # 2. Calculate loss and accuracy
    # BCEWithLogitsLoss calculates loss using logits
    loss = loss_fn(y_logits, y_blob_train)
    train_pred = torch.argmax(y_pred_probs, dim=1)
    acc = accuracy_fn(y_true=y_blob_train, y_pred=train_pred)
    # 3. Optimizer zero grad
    optimizer.zero_grad()
    # 4. Loss backward
    loss.backward()
    # 5. Optimizer step
    optimizer.step()
    # Testing
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_pred_prob = torch.softmax(test_logits, dim=1)
        test_pred = torch.argmax(test_pred_prob, dim=1)
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_pred)

    # Print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.5f} |", end="")
        print(f" Test loss: {test_loss:.5f} | Test acc: {test_acc:.5f}")


model_4.eval()
with torch.inference_mode():
    test_logits = model_4(X_blob_test)
    test_pred_prob = torch.softmax(test_logits, dim=1)
    test_pred = torch.argmax(test_pred_prob, dim=1)

y_blob_test.shape
y_blob_test[:5]
test_pred[:5]

decision_plots(model_4, X_blob_train, y_blob_train, X_blob_test, test_pred)

# classification metrics
# * accuracy
# * precision
# * recall
# * F1 score
# https://lightning.ai/docs/torchmetrics/stable/
# /Users/matt/obsidian/ML/Evaluating\ a\ Classifier.md
# pip install torchmetrics

tm_accuracy = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(device)
tm_accuracy(test_pred, y_blob_test)


