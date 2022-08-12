# Classification
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/02_pytorch_classification.ipynb
# https://youtu.be/Z_ikDlimN6A?t=30690

import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_DIR = "/Users/matt/prog/torch_daniel/models"

# Architectured of a classification problem
# https://learnpytorch.io/02_pytorch_classification/#0-architecture-of-a-classification-neural-network

# Create some sample data
import sklearn
from sklearn.datasets import make_circles

n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

len(X)
len(y)

X[:5]
y[:5]
X.shape

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()

circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
circles

# https://youtu.be/Z_ikDlimN6A?t=33107






