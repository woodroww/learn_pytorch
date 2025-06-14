# https://youtu.be/LyJtbe__2i0?si=g7OlFBltiOCLFumH&t=71862
# data: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/data/pizza_steak_sushi.zip

import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
import random
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch


home_dir = os.path.expanduser('~')

style_file = ".config/matplotlib/stylelib/gruvbox.mplstyle"
style_path = os.path.join(home_dir, style_file)
plt.style.use(style_path)

ml_data_dir = os.path.join(home_dir, "ml_data")
data_dir = os.path.join(ml_data_dir, "pizza_steak_sushi")

train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

train_dir, test_dir

random.seed(42)


def random_image():
    image_path = Path(data_dir)
    image_path_list = list(image_path.glob("*/*/*.jpg"))
    random_image_path = random.choice(image_path_list)
    image_class = random_image_path.parent.stem
    img = Image.open(random_image_path)
    print(f"Random image path: {random_image_path}")
    print(f"Image class: {image_class}")
    print(f"Image height: {img.height}")
    print(f"Image width: {img.width}")
    plt.imshow(img)
    plt.show()


random_image()

data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])


def random_transformed_image():
    image_path = Path(data_dir)
    image_path_list = list(image_path.glob("*/*/*.jpg"))
    random_image_path = random.choice(image_path_list)
    image_class = random_image_path.parent.stem
    img = Image.open(random_image_path)
    print(f"Random image path: {random_image_path}")
    print(f"Image class: {image_class}")
    print(f"Image height: {img.height}")
    print(f"Image width: {img.width}")
    # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
    channel_switch = data_transform(img).permute(1, 2, 0)
    plt.imshow(channel_switch)
    plt.show()


random_transformed_image()

train_data = ImageFolder(train_dir, transform=data_transform)
test_data = ImageFolder(test_dir, transform=data_transform)
class_names = train_data.classes
class_dict = train_data.class_to_idx


def data_preview(i):
    img, label = train_data[i][0], train_data[i][1]
    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")
    img_permute = img.permute(1, 2, 0)
    print(f"Original shape: {img.shape} -> [color_channels, height, width]")
    print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")
    plt.figure(figsize=(10, 7))
    plt.imshow(img.permute(1, 2, 0))
    plt.axis("off")
    plt.title(class_names[label], fontsize=14)
    plt.show()


print(train_data)
len(train_data)
data_preview(52)


