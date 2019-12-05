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

from superres import superres
#from transformer import TransformerNet
from vgg import Vgg16
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def transform(batch):
    for im in batch:
        im = skimage.img_as_float(skimage.io.imread(os.path.join('images', img)))
        im = skimage.transform.resize(skimage.filters.gaussian(im1), (72, 72), order=3)

def train():
    epochs = 80
    lr = 1e-3
    data_path = " "
    batch_size = 4

    mean = [0.485,0.456, 0.406]
    std = [0.229,0.224, 0.225]
    mean = torch.Tensor(mean).reshape(1, -1, 1, 1).to(device)
    std = torch.Tensor(std).reshape(1, -1, 1, 1).to(device)
    def normalize(x):
        x = x.div(255)
        return (x - mean) / std

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x:x.mul(255))
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True
    )

    model_SR = superres().to(device)
    optimizer = torch.optim.Adam(model_SR.parameters(), lr=lr)
    criterion = torch.nn.MSE()

    vgg = Vgg16(requires_grad=False).to(device)

    for e in tqdm(range(epochs)):
        loss = 0
        for idx, (example_data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            example_data = example_data.to(device)
            output = model_SR(example_data)
            output = normalize(output)





if __name__ == '__main__':
    transform()
    # train()
