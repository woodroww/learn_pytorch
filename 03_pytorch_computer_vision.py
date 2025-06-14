# ------------------------------------------------------------------------------
# Computer Vision
# ------------------------------------------------------------------------------
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/03_pytorch_computer_vision.ipynb
# https://youtu.be/LyJtbe__2i0?si=8jJSop4uj0uI2kys&t=53497

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from helper_functions import accuracy_fn
from helper_functions import print_train_time
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import random
import numpy as np
import os

print(torch.__version__)
print(torchvision.__version__)
device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
plt.ion()

# ------------------------------------------------------------------------------
# getting a dataset
# ------------------------------------------------------------------------------
# https://docs.pytorch.org/vision/stable/datasets.html

train_data = torchvision.datasets.FashionMNIST(
    '/Users/matt/prog/ml/daniel/data/',
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None)

test_data = torchvision.datasets.FashionMNIST(
    '/Users/matt/prog/ml/daniel/data/',
    train=False,
    download=True,
    transform=ToTensor())


train_data.data.shape
train_data.data[0]
train_data.classes
class_to_idx = train_data.class_to_idx
class_to_idx
train_data.targets

image, label = train_data[0]
image.shape, label
train_data.classes[label]
image
class_names = train_data.classes

test_data.data.shape


image, label = train_data[0]
print(f"Image shape: {image.shape}")
# image shape is [1, 28, 28] (colour channels, height, width)
plt.imshow(image.squeeze(), cmap="gray")
plt.title(label)
plt.show()

fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(train_data.classes[label])
    plt.axis(False)

# Right now, our data in the form of PyTorch Datasets.
# DataLoader turns our dataset into a Python iterable.
# We need batches for reasonable memory usage
# Batches also give the model more chances to update

BATCH_SIZE = 32
train_dataloader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_dataloader = DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

# What have we created?

print(
    f"length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(
    f"length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape, train_labels_batch.shape

torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
image, label = train_features_batch[random_idx], train_labels_batch[random_idx]

plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)
plt.show()
print(f"Image size: {img.shape}")
print(f"Label: {label}, label size: {label.shape}")


# ------------------------------------------------------------------------------
# Baseline Model
# ------------------------------------------------------------------------------
# https://youtu.be/LyJtbe__2i0?si=zME4G-hCQLOmhYzV&t=56155

flatten_model = nn.Flatten()
x = train_features_batch[0]
output = flatten_model(x)
print(f"Shape before flattening: {x.shape}")
print(f"Shape after flattening:  {output.shape}")


class FashionMNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape,
                 hidden_units,
                 output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)


torch.manual_seed(42)
output.shape[1]
model_0 = FashionMNISTModelV0(
    output.shape[1], 10, len(train_data.classes)).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)


model_0.eval()
with torch.inference_mode():
    y_logits = model_0(train_features_batch.to(device))
y_pred_probs = torch.softmax(y_logits, dim=1)
y_preds = torch.argmax(y_pred_probs, dim=1)

X, y = next(iter(train_dataloader))
model_0.train()
y_pred = model_0(X.to(device))
y_pred.shape, y.shape

torch.manual_seed(42)
start_time = timer()
model_0.to(device)
epochs = 3
for epoch in tqdm(range(epochs)):
    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        X = X.to(device)
        y = y.to(device)
        y_pred = model_0(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 400 == 0:
            samples = batch * len(X)
            total_samples = len(train_dataloader.dataset)
            print(f"Looked at {samples}/{total_samples} samples")
    train_loss /= len(train_dataloader)
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for (X, y) in test_dataloader:
            X = X.to(device)
            y = y.to(device)
            test_pred = model_0(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    print(f"Epoch: {epoch} | Loss: {train_loss:.5f}, | Test Loss:", end="")
    print(f" {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
end_time = timer()
total_train_time_model_0 = print_train_time(start=start_time,
                                            end=end_time,
                                            device=device)

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)

            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))  # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,  # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


# Calculate model 0 results on test dataset
model_0_results = eval_model(model=model_0, data_loader=test_dataloader,
                             loss_fn=loss_fn, accuracy_fn=accuracy_fn
                             )
model_0_results


# ------------------------------------------------------------------------------
# Create a model with non-linear and linear layers
# ------------------------------------------------------------------------------

class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # flatten inputs into single vector
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)


torch.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape=784,  # number of input features
                              hidden_units=10,
                              # number of output classes desired
                              output_shape=len(class_names)
                              ).to(device)  # send model to GPU if it's available

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)


# ------------------------------------------------------------------------------
# Functionizing training and test loops
# ------------------------------------------------------------------------------

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))  # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                    # Go from logits -> pred labels
                                    y_pred=test_pred.argmax(dim=1)
                                    )

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


torch.manual_seed(42)
train_time_start_on_gpu = timer()
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader,
               model=model_1,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn
               )
    test_step(data_loader=test_dataloader,
              model=model_1,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn
              )

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)

model_1_results = eval_model(model=model_1, data_loader=test_dataloader,
                             loss_fn=loss_fn, accuracy_fn=accuracy_fn,
                             device=device
                             )

model_0_results, model_1_results

# So baseline model is better

# ------------------------------------------------------------------------------
# Convolutional Neural Network
# ------------------------------------------------------------------------------
# https://youtu.be/LyJtbe__2i0?si=lwkJAigWipwDQRPO&t=63552

# Create a convolutional neural network


class FashionMNISTModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,  # how big is the square that's going over the image?
                      stride=1,  # default
                      padding=1),  # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)  # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the
            # shape of our input data. Use the forward pass print statements to
            # determine or calculate it yourself 
            # see Shape: in https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(f"output shape of conv_block_1 {x.shape}")
        x = self.block_2(x)
        # print(f"output shape of conv_block_2 {x.shape}")
        x = self.classifier(x)
        # print(f"output shape of classifier {x.shape}")
        return x


torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)
model_2

# Stepping through nn.Conv2d()
# make some dummy data
torch.manual_seed(42)

images = torch.randn(size=(32, 3, 64, 64))
test_image = images[0]

test_image.shape

# create a single conv2d layer
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=0)

conv_output = conv_layer(test_image)
conv_output.shape

image.shape
plt.imshow(image[0])

output = model_2(image.to(device).unsqueeze(0))
output.shape


# output shape of conv_block_1 torch.Size([1, 10, 14, 14])
# output shape of conv_block_2 torch.Size([1, 10, 7, 7])
# output shape of classifier torch.Size([1, 10])
test_flatten = torch.flatten(torch.randn(size=(1, 10, 7, 7)))
test_flatten.shape

# https://youtu.be/LyJtbe__2i0?si=lNIhQWSc0J5EhnTh&t=66482

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

torch.manual_seed(42)
train_time_start_on_gpu = timer()
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader,
               model=model_2,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn
               )
    test_step(data_loader=test_dataloader,
              model=model_1,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn
              )

train_time_end_on_gpu = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)
model_2_results = eval_model(model=model_2, data_loader=test_dataloader,
                             loss_fn=loss_fn, accuracy_fn=accuracy_fn
                             )

compare_results = pd.DataFrame([
    model_2_results, model_1_results, model_0_results])
compare_results["training_time"] = [
    total_train_time_model_2, total_train_time_model_1, total_train_time_model_0]

print(compare_results)

def make_predictions(model, data, device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs)

test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)
    
plt.imshow(test_samples[0].squeeze(), cmap='gray')
plt.title(class_names[test_labels[0]])

pred_probs = make_predictions(model_2, test_samples, device)
pred_labels = pred_probs.argmax(dim=1)
pred_labels

len(test_labels), len(pred_labels), len(class_names)


# Plot predictions
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    # Create a subplot
    plt.subplot(nrows, ncols, i+1)
    # Plot the target image
    plt.imshow(sample.squeeze(), cmap="gray")
    # Find the prediction label (in text form, e.g. "Sandal")
    pred_label = class_names[pred_labels[i]]
    # Get the truth label (in text form, e.g. "T-shirt")
    truth_label = class_names[test_labels[i]]
    # Create the title text of the plot
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"
    # Check for equality and change title colour accordingly
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g") # green text if correct
    else:
        plt.title(title_text, fontsize=10, c="r") # red text if wrong
    plt.axis(False)



# https://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/
from mlxtend.plotting import plot_confusion_matrix

binary1 = np.array([[4, 1], [1, 2]])

fig, ax = plot_confusion_matrix(conf_mat=binary1)
plt.show()

import torchmetrics, mlxtend
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

y_preds = []
model_2.eval()
with torch.inference_mode():
  for X, y in tqdm(test_dataloader, desc="Making predictions"):
    # Send data and targets to target device
    X, y = X.to(device), y.to(device)
    # Do the forward pass
    y_logit = model_2(X)
    # Turn predictions from logits -> prediction probabilities -> predictions labels
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
    # Put predictions on CPU for evaluation
    y_preds.append(y_pred.cpu())
# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)

# 2. Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
    class_names=class_names, # turn the row and column labels into class names
    figsize=(10, 7)
)

# We can see our model does fairly well since most of the dark squares are down
# the diagonal from top left to bottom right (and ideal model will have only
# values in these squares and 0 everywhere else).
# The model gets most "confused" on classes that are similar, for example
# predicting "Pullover" for images that are actually labelled "Shirt".
# And the same for predicting "Shirt" for classes that are actually labelled
# "T-shirt/top".
# This kind of information is often more helpful than a single accuracy metric
# because it tells use *where* a model is getting things wrong.
# It also hints at *why* the model may be getting certain things wrong.
# It's understandable the model sometimes predicts "Shirt" for images labelled
# "T-shirt/top".
# We can use this kind of information to further inspect our models and data to
# see how it could be improved.
# > **Exercise:** Use the trained `model_2` to make predictions on the test FashionMNIST dataset. Then plot some predictions where the model was wrong alongside what the label of the image should've been. After visualizing these predictions do you think it's more of a modelling error or a data error? As in, could the model do better or are the labels of the data too close to each other (e.g. a "Shirt" label is too close to "T-shirt/top")?

home_dir = os.path.expanduser('~')
model_dir = os.path.join(home_dir, "prog/ml/daniel/models")

# Create model save path
model_save_path = os.path.join(model_dir, "03_pytorch_computer_vision_model_2.pth")

# Save the model state dict
print(f"Saving model to: {model_save_path}")
torch.save(obj=model_2.state_dict(), # only saving the state_dict() only saves the learned parameters
           f=model_save_path)


loaded_model_2 = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)

loaded_model_2.load_state_dict(torch.load(f=model_save_path))

loaded_model_2_results = eval_model(
    model=loaded_model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn, 
    accuracy_fn=accuracy_fn
)

loaded_model_2_results, model_2_results

torch.isclose(torch.tensor(loaded_model_2_results["model_loss"]),
              torch.tensor(model_2_results["model_loss"]))


