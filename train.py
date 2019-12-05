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
import cv2
import matplotlib.animation as animation

from generator import Generator
#from transformer import TransformerNet
from vgg import Vgg16
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def gram(input):
    nb, nch, h, w = input.shape
    features = input.view(nb, nch, h*w)
    features_t = features.transpose(1,2)
    gram = features.bmm(features_t/(nch*h*w))
    '''
    for im in range(num):.b
        t = t.reshape(channel, h*w)
        g = np.matmul(t, t.T) / (channel * h * w)
        grams.append(g)
    gr = torch.Tensor(grams)
    '''
    return gram
        

def train(restore_path=None):
    epochs = 1
    lr = 1e-3
    content_weight = 1e5
    style_weight = 1e10
    batch_size = 4
    data_path = 'COCO/'
    mean = [0.485,0.456, 0.406]
    std = [0.229,0.224, 0.225]
    mean = torch.Tensor(mean).reshape(1, -1, 1, 1).to(device)
    std = torch.Tensor(std).reshape(1, -1, 1, 1).to(device)
    def normalize(x):
        x = x.div(255)
        return (x - mean) / std
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((256, 256)),
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
        shuffle=True,
        pin_memory=True
    )

    gen = Generator().to(device)
    if restore_path is not None:
        gen.load_state_dict(torch.load(restore_path))
    optimizer = torch.optim.Adam(gen.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)

    # compute style features and gram matrix
    style_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x:x.mul(255))
    ])
    style = Image.open("rickmorty-style.jpg")
    style = style_transform(style)
    style = style.to(device)
    style = normalize(style)
    style = style.repeat(batch_size, 1, 1, 1)
    features_style = vgg(style)
    gr_norm = [gram(ft) for ft in features_style]
    for e in tqdm(range(epochs)):
        loss = 0
        agg_content_loss = 0
        agg_style_loss = 0
        avg_time = 0
        for idx, (example_data, _) in enumerate(train_loader):
            start = time.time()
            optimizer.zero_grad()
            example_data = example_data.to(device)
            output = gen(example_data)
            output = normalize(output)
            example_data = normalize(example_data)
            features_output = vgg(output)
            features_content = vgg(example_data)
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
                        e + 1, (idx + 1)*4, len(train_dataset),
                        agg_content_loss / (idx + 1),
                        agg_style_loss / (idx + 1),
                        (agg_content_loss + agg_style_loss) / (idx + 1)
                    )
                tqdm.write(mesg)
            if idx % 2000 == 1999:
                torch.save(gen.state_dict(), f'models/gen_ep{e}_b{idx}.pt')
            optimizer.step()


    torch.save(gen.state_dict(), f'gen.pt')
    test = Image.open("COCO/data/2015-07-19 20:28:53.jpg")
    test = test.transpose(1, 3, 256, 256)
    stylized = gen(test)[0].detach().cpu.numpy()
    stylized = stylized.transpose(1, 2, 0)
    plt.imshow(stylized)
    plt.show()

def test():
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(f'models/rain-princess_gen_all.pt', map_location=torch.device('cpu')))
    gen.eval()
    test = Image.open("grr.jpg")
    test = test.resize((256, 256))
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()]
    )
    test = transform(test)
    test = test[:3]
    print(test.shape)
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


def convert(video_path):
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(f'models/mosaic_gen_all.pt', map_location=torch.device('cpu')))
    gen.eval()

    vidcap = cv2.VideoCapture(video_path)
    success,test = vidcap.read()
    count = 0
    transform=torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((360, 360)), 
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.ToTensor()]
    )

    converted_video_frames = []
    while success:
        test = Image.fromarray(test*255)
        # test = test.resize((256, 256))
        test = transform(test) 
        print('Read a new frame: ', success)
        count += 1
        test = test.reshape(1, 3, 256, 256)
        test = (test - torch.min(test))
        test = test / torch.max(test)
        test = test * 255
        stylized = gen(test)[0].detach().cpu().numpy()
        arr = stylized.transpose(1, 2, 0)
        new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min())))
        converted_video_frames.append(new_arr)

        success,test = vidcap.read()
    print("done stylizing")
    frames = [] # for storing the generated images
    fig = plt.figure()
    for i in range(len(converted_video_frames)):
        frames.append([plt.imshow(converted_video_frames[i], animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=33.33, blit=True,
                                    repeat_delay=1000)
    plt.axis('off')
    ani.save('mosaic_dogvid.mp4')
    plt.show()


if __name__ == '__main__':
    restore_path = 'models/gen_ep0_b19999.pt'
    #train(restore_path)
    #test()
    convert('videos/dogvid.mp4')
