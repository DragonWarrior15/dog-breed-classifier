import os
import logging

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
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
        self.img_transform = transforms.Compose([
                                transforms.RandomAffine(degrees=15,
                                                        translate=(0.1, 0.1),
                                                        scale=(0.9, 1.1),
                                                        shear=(-10, 10)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.Resize((resize_size, resize_size)),
                                transforms.Lambda(lambda x: x/255.0)
                             ])

        self.img_transform_test = transforms.Compose([
                                    transforms.Resize((resize_size, resize_size)),
                                    transforms.Lambda(lambda x: x/255.0)
                                  ])

# dataset class
class dogBreedDataset(Dataset):
    """we will have a map style dataset, need to define init, len and getitem
    the iterable will be a list of tuples where the first item of tuple
    is the class, second is the label, and third is the path of the file"""
    def __init__(self, img_dir, img_transform=None, target_transform=None,
                 limit_classes=False):
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

            # in case limit classes is on, we only keep images for a few classes
            if(limit_classes and img_class > 9):
                continue
            else:
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
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x, debug=False):
        if(debug):
            for layer in self.nn_stack:
                x = layer(x)
                print(x.size())
            logits = x
        else:
            logits = self.nn_stack(x)
        return logits

def logger_setup(log_file_path):
    # define logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO) # set the level of logger first
    # f_handler = logging.StreamHandler('logs')
    f_handler = logging.FileHandler(log_file_path)
    f_handler.setLevel(logging.INFO)
    # f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return logger
