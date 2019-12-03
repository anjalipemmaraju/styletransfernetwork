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

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
from PIL import Image

from generator import Generator
from vgg import Vgg16
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def normalize(x):
    mean = [0.406, 0.485,0.456]
    std = [0.225, 0.229,0.224]
    mean = torch.Tensor(mean).reshape(1, -1, 1, 1)
    std = torch.Tensor(std).reshape(1, -1, 1, 1)
    x = (x - mean) / std
    return x

def gram(input):
    grams = []
    num, channel, h, w = input.shape
    for im in range(num):
        t = input[im].detach().cpu().numpy()
        t = t.reshape(channel, h*w)
        g = np.matmul(t, t.T) / (channel * h * w)
        grams.append(g)
    gr = torch.Tensor(grams)
    return gr
        

def train():
    epochs = 2
    lr = 10e-3
    content_weight = 1e-5
    style_weight = 5e-10
    batch_size = 4
    data_path = 'COCO/'
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.Resize(256, 256),
        torchvision.transforms.CenterCrop((256, 256)),
        torchvision.transforms.ToTensor(),
        ])
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=16,
        shuffle=True
    )

    gen = Generator().to(device)
    optimizer = torch.optim.Adam(gen.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)

    style = Image.open("rickmorty-style.jpg")
    style = style.resize((256, 256))
    style = transform(style).to(device)
    normalized_style = normalize(style.reshape(1, 3, style.shape[1], style.shape[2]))
    normalized_style = (normalized_style - torch.min(normalized_style))
    normalized_style = normalized_style / torch.max(normalized_style)

    normalized_style = normalized_style.repeat(batch_size, 1, 1, 1).to(device)
    features_style = vgg(normalized_style)

    gr_norm = [gram(ft) for ft in features_style]
    for e in tqdm(range(epochs)):
        loss = 0
        for idx, (example_data, _) in enumerate(train_loader):
            style_loss = 0
            optimizer.zero_grad()
            output = gen(example_data)
            normalized_output = normalize(output)
            features_x = vgg(normalized_output)

            feature_loss = content_weight * criterion(features_style.relu2_2, features_x.relu2_2)

            for ft, gr_style in zip(features_x, gr_norm):
                gr = gram(ft)
                style_loss += criterion(gr, gr_style)
            style_loss = style_weight * style_loss
            loss = feature_loss + style_loss
            loss.backward()
            optimizer.step()

    torch.save(gen.state_dict(), f'gen.pt')
    test = Image.open("COCO/data/2015-07-19 20:28:53.jpg")
    test = test.resize((256, 256))
    test = torch.Tensor(test).transpose(1, 3, 256, 256)
    stylized = gen(test)[0].detach().cpu.numpy()
    stylized = stylized.transpose(1, 2, 0)
    plt.imshow(stylized)
    plt.show()

if __name__ == '__main__':
    train()
