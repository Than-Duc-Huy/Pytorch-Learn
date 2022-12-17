import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import time
import os 
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25,0.25])


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ]),
    'val': transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])
}

# Import data in the folder
DATA_DIR = "./data"
sets = ['train', 'val']

image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),data_transforms[x]) for x in sets}
# Access using image_datasets['train']


BATCH_SIZE = 4

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = BATCH_SIZE, shuffle = True, num_workers = 0) for x in sets}


dataset_sizes = {x: len(image_datasets[x]) for x in sets}
class_name = image_datasets['train'].classes

# print(type(image_datasets['train']))
# print(dir(image_datasets['train']))

# sample = iter(dataloaders['train'])
# print(sample)
# print(type(sample))
# print(sample.next())
# x, y = sample.next()
# print(x.shape, y.shape)

print(class_name)


model = models.resnet18(pretrained=True)
print(type(model))
print(model)
print(dir(model))

### Freeze update
for param in model.parameters():
    param.requires_grad = False # Freeze all


num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)  # When define a new layer, the grad is automatically True
model.to(device)

