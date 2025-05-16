# ------------------------------------------------------------------------------
# Computer Vision
# ------------------------------------------------------------------------------
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/03_pytorch_computer_vision.ipynb
# https://youtu.be/LyJtbe__2i0?si=8jJSop4uj0uI2kys&t=53497

import torch
from torch import nn

import torchvision

# from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
from helper_functions import accuracy_fn
from helper_functions import print_train_time
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

print(torch.__version__)
print(torchvision.__version__)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
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

print(f"length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

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
model_0 = FashionMNISTModelV0(output.shape[1], 10, len(train_data.classes)).to(device)
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
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")
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
    print(f"Epoch: {epoch} | Loss: {train_loss:.5f}, | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
end_time = timer()
print(f"{end_time - start_time:.3f} seconds")




