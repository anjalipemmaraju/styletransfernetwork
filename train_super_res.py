import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import time

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
from PIL import Image

from superres import SuperResolution
#from transformer import TransformerNet
from vgg import Vgg16
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def transform(data_path):
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform
    )


def train():
    epochs = 80
    lr = 1e-3
    data_path = "data"
    batch_size = 4

    mean = [0.485,0.456, 0.406]
    std = [0.229,0.224, 0.225]
    mean = torch.Tensor(mean).reshape(1, -1, 1, 1).to(device)
    std = torch.Tensor(std).reshape(1, -1, 1, 1).to(device)
    def normalize(x):
        x = x.div(255)
        return (x - mean) / std

    coarse_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((72,72)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x:x.mul(255))
    ])
    fine_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((288,288)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x:x.mul(255))
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True
    )

    
    model_SR = SuperResolution().to(device)
    optimizer = torch.optim.Adam(model_SR.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)

    for e in tqdm(range(epochs)):
        loss = 0
        for idx, (data, _) in enumerate(train_loader):
            if idx < 2500:
                coarse = coarse_transform(data.clone())
                coarse = coarse.to(device)
                fine = fine_transform(data.clone())
                fine = fine.to(device)
                fine = normalize(fine)
                optimizer.zero_grad()
                coarse_output = model_SR(coarse)
                coarse_output = normalize(coarse_output)
                coarse_features = vgg(coarse_output)
                fine_features = vgg(fine)
                loss = criterion(coarse_features.relu2_2, fine_features.relu2_2)
                loss.backward()
                optimizer.step()
                if idx % 100 == 0:
                    tqdm.write(f'epoch {e} \t batch {idx} \t loss = {loss.item()}')
        torch.save(model_SR.state_dict(), f'models/SR_ep{e}.pt')

if __name__ == '__main__':
    #transform()
     train()
