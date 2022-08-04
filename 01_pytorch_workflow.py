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



