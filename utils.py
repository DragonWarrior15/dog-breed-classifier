import os

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Compose, Lambda
from torchvision.io import read_image
from torch import nn

# doesnt work
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# definition for functions to work with data, the hierarchy is
# dogImages
# | - train
# |     | - 001.Affenpinscher
# |     | - 002.Afghan_hound
# | - val
# |    | - 001.Affenpinscher
# | - test
# |    | - 001.Affenpinscher

# device for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class dogBreedTransforms():
    def __init__(self, resize_size=28):
        self.img_transform = Compose([Resize((resize_size, resize_size)),
                                      Lambda(lambda x: x/255.0)])

# dataset class
class dogBreedDataset(Dataset):
    """we will have a map style dataset, need to define init, len and getitem
    the iterable will be a list of tuples where the first item of tuple
    is the class, second is the label, and third is the path of the file"""
    def __init__(self, img_dir, img_transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_transform = img_transform
        self.target_transform = target_transform
        # the iterable to store the (class, label, path) pairs
        self.img_list = []
        # a set to store all the classes
        self.all_classes = set()
        # walk through the directories
        for di in os.listdir(self.img_dir):
            img_class = int(di[:3]) - 1
            img_label = di[4:]
            if img_class not in self.all_classes:
                self.all_classes = self.all_classes.union(set([img_class]))
            for f in os.listdir(os.path.join(self.img_dir, di)):
                self.img_list.append((img_class, img_label,
                                        os.path.join(self.img_dir, di, f)))

    def get_total_classes(self):
        return len(self.all_classes)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # read the image
        img = read_image(self.img_list[idx][2]).float()
        target = self.img_list[idx][0]
        # apply any transforms
        if self.img_transform:
            img = self.img_transform(img)
        # apply target transform
        if self.target_transform:
            target = self.target_transform

        return img, target

class dogBreedClassifier(nn.Module):
    def __init__(self, input_size, output_size=2):
        super(dogBreedClassifier, self).__init__()
        self.input_size = input_size
        self.nn_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Flatten(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        # for layer in self.nn_stack:
            # x = layer(x)
            # print(x.size())
        # logits = x
        logits = self.nn_stack(x)
        return logits
