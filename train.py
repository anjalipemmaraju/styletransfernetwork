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
from transformer import TransformerNet
from vgg import Vgg16
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
def normalize(x):
    mean = [0.485,0.456, 0.406]
    std = [0.229,0.224, 0.225]
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
    lr = 1e-3
    content_weight = 1e5
    style_weight = 1e10
    batch_size = 4
    data_path = 'COCO/'
    image_normalize = torchvision.transforms.Normalize(
            mean=[0.485,0.456, 0.406],
            std = [0.229,0.224, 0.225]
        ) 
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.Resize(256, 256),
        torchvision.transforms.CenterCrop((256, 256)),
        torchvision.transforms.ToTensor(),
        image_normalize
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

    # compute style features and gram matrix
    style = Image.open("rickmorty-style.jpg")
    style = style.resize((256, 256))
    normalized_style = transform(style)
    normalized_style = normalized_style.reshape(1, 3, normalized_style.shape[1], normalized_style.shape[2])
    normalized_style = normalized_style.repeat(batch_size, 1, 1, 1)
    normalized_style = normalized_style.to(device)
    features_style = vgg(normalized_style)
    gr_norm = [gram(ft) for ft in features_style]
    for e in tqdm(range(epochs)):
        loss = 0
        agg_content_loss = 0
        agg_style_loss = 0
        for idx, (example_data, _) in enumerate(train_loader):
            #tqdm.write(f'batch idx: {idx}')
            optimizer.zero_grad()
            example_data = example_data * 255
            output = gen(example_data.to(device))
            normalized_output = output/255.0
            features_output = vgg(normalized_output)
            features_content = vgg(normalize(example_data))
            content_loss = content_weight * criterion(features_output.relu2_2, features_content.relu2_2)
            style_loss = 0
            for ft, gr_style in zip(features_output, gr_norm):
                gr = gram(ft)
                style_loss += criterion(gr, gr_style)
            style_loss = style_weight * style_loss
            loss = content_loss + style_loss
            loss.backward()
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            if idx % 500 == 0:
                mesg = "Epoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                        e + 1, idx + 1, len(train_dataset),
                        agg_content_loss / (idx + 1),
                        agg_style_loss / (idx + 1),
                        (agg_content_loss + agg_style_loss) / (idx + 1)
                    )
                tqdm.write(mesg)
            optimizer.step()

    torch.save(gen.state_dict(), f'gen.pt')
    test = Image.open("COCO/data/2015-07-19 20:28:53.jpg")
    test = test.resize((256, 256))
    test = torchvision.transforms.ToTensorTensor(test)
    test = test.transpose(1, 3, 256, 256)
    stylized = gen(test)[0].detach().cpu.numpy()
    stylized = stylized.transpose(1, 2, 0)
    plt.imshow(stylized)
    plt.show()

def test():
    gen = TransformerNet().to(device)
    gen.load_state_dict(torch.load(f'models/gen.pt', map_location=torch.device('cpu')))
    gen.eval()
    test = Image.open("COCO/data/2010-08-10 00:15:25.jpg")
    test = test.resize((256, 256))
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    test = transform(test)
    test = test.reshape(1, 3, 256, 256)
    test = (test - torch.min(test))
    test = test / torch.max(test)
    test = test * 255
    stylized = gen(test)[0].detach().cpu().numpy()
    arr = stylized.transpose(1, 2, 0)
    new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min())))
    print(new_arr)
    print(np.amin(new_arr))
    print(np.amax(new_arr))
    plt.imshow(new_arr)
    plt.show()

if __name__ == '__main__':
    train()
    #test()