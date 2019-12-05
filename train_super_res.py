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


def train(start_epoch=-1):
    epochs = 80
    lr = 1e-3
    data_path = "COCO"
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

    fine_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=fine_transform
    )

    coarse_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=coarse_transform
    )

    coarse_loader = torch.utils.data.DataLoader(
        coarse_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True
    )

    fine_loader = torch.utils.data.DataLoader(
            fine_dataset,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True
    )   

    
    model_SR = SuperResolution().to(device)
    if start_epoch != -1:
        model_SR.load_state_dict(torch.load(f'models/SR_ep{start_epoch}.pt'))
    optimizer = torch.optim.Adam(model_SR.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)

    for e in tqdm(range(start_epoch+1, epochs)):
        loss = 0
        total_loss = 0
        for idx, (coarse_data, fine_data),  in enumerate(zip(coarse_loader, fine_loader)):
            coarse_data = coarse_data[0]
            fine_data = fine_data[0]
            #coarse = coarse_transform(coarse_data.clone())
            coarse = coarse_data.to(device)
            fine = fine_data.to(device)
            fine = normalize(fine)
            optimizer.zero_grad()
            coarse_output = model_SR(coarse)
            coarse_output = normalize(coarse_output)
            coarse_features = vgg(coarse_output)
            fine_features = vgg(fine)
            #print(fine_features.relu2_2.shape)
            #print(coarse_features.relu2_2.shape)
            loss = criterion(coarse_features.relu2_2, fine_features.relu2_2)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            if idx % 100 == 0:
                tqdm.write(f'epoch {e} \t batch {idx} \t avg batch loss = {total_loss/(idx+1)}')
        torch.save(model_SR.state_dict(), f'models/SR_ep{e+1}.pt')

def test():
    gen = SuperResolution().to(device)
    gen.load_state_dict(torch.load(f'models/SR_ep63.pt', map_location=torch.device('cpu')))
    gen.eval()
    test = Image.open("nature.jpg")
    test = test.resize((256, 256))
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()]
    )
    test = transform(test)
    test = test.reshape(1, 3, 256, 256)
    test = (test - torch.min(test))
    test = test / torch.max(test)
    test = test * 255
    stylized = gen(test)[0].detach().cpu().numpy()
    arr = stylized.transpose(1, 2, 0)
    new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min())))
    print(new_arr.max())
    plt.imshow(new_arr)
    plt.show()
if __name__ == '__main__':
    #transform()
    #train(start_epoch=1)
    test()
