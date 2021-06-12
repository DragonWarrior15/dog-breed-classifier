import os
from datetime import datetime

from utils import dogBreedDataset, dogBreedTransforms, dogBreedClassifier, logger_setup

import torch
from torch.utils.data import DataLoader
from torch import nn

log_dir = 'logs'
model_dir = 'models'

# setup logging and file names
dt = datetime.now().strftime('%Y%m%d%H%M')
if not os.path.exists('logs'):
    os.mkdir('logs')
logger = logger_setup(os.path.join('logs', f'{dt}'))

logger.info(f'logging to file {os.path.join(log_dir, f"{dt}")}')

# make any directories for saving models
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(os.path.join(model_dir, f'{dt}')):
    os.mkdir(os.path.join(model_dir, f'{dt}'))

logger.info(f'saving models to folder {os.path.join(model_dir, f"{dt}")}')


learning_rate = 1e-3
batch_size = 128
epochs = 100
input_size = 32
limit_classes = True
optimizer_class = torch.optim.Adam

logger.info(f'learning rate is {learning_rate}')
logger.info(f'batch_size is {batch_size}')
logger.info(f'epochs is {epochs}')
logger.info(f'input image size is {input_size}')
logger.info(f'limit_classes is set to {limit_classes}')
logger.info(f'optimizer is {optimizer_class}')

transforms = dogBreedTransforms(resize_size=input_size)

logger.info(f'image transforms in use\n {transforms.img_transform}')

training_data = dogBreedDataset('data/dogImages/train',
                                img_transform=transforms.img_transform,
                                limit_classes=limit_classes)
train_dataloader = DataLoader(training_data, batch_size=batch_size,
                                shuffle=True)

validation_data = dogBreedDataset('data/dogImages/valid',
                                  img_transform=transforms.img_transform,
                                  limit_classes=limit_classes)
validation_dataloader = DataLoader(validation_data, batch_size=len(validation_data),
                                    shuffle=False)

num_classes = training_data.get_total_classes()
logger.info(f'total classes are {num_classes}')

# train_dataloader = DataLoader(training_data, batch_size=batch_size,
                                # sampler=list(range(batch_size)))

model = dogBreedClassifier(input_size=input_size, output_size=num_classes)

logger.info(model)

loss_fn = nn.CrossEntropyLoss()

optimizer = optimizer_class(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # zero out gradients first
        optimizer.zero_grad()

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
        # if True:
            loss, current = loss.item(), batch * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def val_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    logger.info(f"Validation Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    logger.info(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    val_loop(validation_dataloader, model, loss_fn)
    # save the model
    torch.save(model.state_dict(), os.path.join(model_dir, f'{dt}', '%05d' % t))

logger.info("Done!")
