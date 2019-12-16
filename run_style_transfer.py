import torch
import glob
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
from vgg import Vgg16
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
# mean and std of vgg trained input
mean = [0.485,0.456, 0.406]
std = [0.229,0.224, 0.225]
''' gram matrix function
Input:
    input (torch tensor): data to take the gram matrix of
Output:
    gram (matrix): gram matrix of input data
'''
def gram(input):
    nb, nch, h, w = input.shape
    features = input.view(nb, nch, h*w)
    features_t = features.transpose(1,2)
    gram = features.bmm(features_t/(nch*h*w))
    return gram

''' normalize input data based on the vgg train data discribution
Input:
    x (torch tensor): data to be normalized
Output:
    normalized data (torch tensor)
'''
def normalize(x):
    global mean
    global std
    x = x.div(255)
    return (x - mean) / std

''' function to train the generator
Input:
    restore_path (string): path to old model that you want to keep training
Output:
    None
'''
def train(restore_path=None):
    epochs = 1
    lr = 1e-3
    content_weight = 1e5
    style_weight = 1e10
    batch_size = 4

    # not downloaded to Git because it is too big, download from COCO website
    data_path = 'COCO/'

    global mean
    global std
    mean = torch.Tensor(mean).reshape(1, -1, 1, 1).to(device)
    std = torch.Tensor(std).reshape(1, -1, 1, 1).to(device)

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

    # load data after transforming to correct size and pixel values
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

    # open style image and turn into batch-sized tensor
    style = Image.open("style_images/rickmorty-style.jpg")
    style = style_transform(style)
    style = style.to(device)
    style = normalize(style)
    style = style.repeat(batch_size, 1, 1, 1)
    features_style = vgg(style)

    # compute gram of vgg output of style image tensor
    gr_norm = [gram(ft) for ft in features_style]

    # start train loop
    for e in tqdm(range(epochs)):
        loss = 0
        agg_content_loss = 0
        agg_style_loss = 0
        avg_time = 0
        for idx, (example_data, _) in enumerate(train_loader):
            start = time.time()
            optimizer.zero_grad()
            example_data = example_data.to(device)

            # pass the output through the generator and normalize
            output = gen(example_data)
            output = normalize(output)

            # normalize the original data
            example_data = normalize(example_data)

            # pass the output and the original data through vgg
            features_output = vgg(output)
            features_content = vgg(example_data)

            # calculate content and style loss as described in the paper
            content_loss = content_weight * criterion(features_output.relu2_2, features_content.relu2_2)
            style_loss = 0
            for ft, gr_style in zip(features_output, gr_norm):
                gr = gram(ft)
                style_loss += criterion(gr, gr_style)
            style_loss = style_weight * style_loss

            # propagate the loss
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

    # save the final model
    torch.save(gen.state_dict(), f'gen.pt')

''' load a model and use it to stylize an image
Input:
    None
Output:
    stylized image (3xhxw numpy array)
'''
def test():
    # load the model in eval mode
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(f'models/vangogh_gen_all.pt', map_location=torch.device('cpu')))
    gen.eval()

    # open the test image and transform into expected-size input for generator
    test = Image.open("content_images/original_dog.jpg")
    test = test.resize((256, 256))
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()]
    )
    test = transform(test)

    # reshape the transformed test image back into a numpy array
    test = test[:3]
    test = test.reshape(1, 3, 256, 256)

    # shift pixel values to lie between 0 and 255
    test = (test - torch.min(test))
    test = test / torch.max(test)
    test = test * 255

    # display stylized image with correct hxwxc format for matplotlib
    stylized = gen(test)[0].detach().cpu().numpy()
    arr = stylized.transpose(1, 2, 0)
    new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min())))
    plt.imshow(new_arr)
    plt.show()

''' Convert a video specified into a stylized video
Input:
    vieo_path (string): path to the video to be stylized
Output:
    stylized video
'''
def convert(video_path):
    gen = Generator().to(device)
    styles = ['rm', 'rain-princess', 'vangogh', 'mosaic']

    # load each style's generator and run it on the video
    for style in styles:
        gen.load_state_dict(torch.load(f'models/{style}_gen_all.pt', map_location=torch.device('cpu')))
        gen.eval()

        # transform the video into the expected input size
        vidcap = cv2.VideoCapture(video_path)
        success,test = vidcap.read()
        count = 0
        transform=torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((360, 360)), 
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.ToTensor()]
        )

        converted_video_frames = []
        # read in each frame from the video
        while success:
            # transform the frame into a tensor of the expected pixel range and shape
            test = Image.fromarray(test*255)
            test = transform(test) 
            count += 1
            test = test.reshape(1, 3, 256, 256)
            test = (test - torch.min(test))
            test = test / torch.max(test)
            test = test * 255

            #stylize the frame and convert it back to numpy
            stylized = gen(test)[0].detach().cpu().numpy()
            arr = stylized.transpose(1, 2, 0)
            new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min())))
            converted_video_frames.append(new_arr)

            success,test = vidcap.read()
        # save all the frames in a npy file for easier reading
        print("done stylizing")
        np.save(f'{style}_frames.npy', converted_video_frames)
    
    # load back frames from the npy files
    for style in styles:
        frames = np.load(f'{style}_frames.npy')
        out = cv2.VideoWriter(f'{style}_dogvid.avi', 0, 30, (256,256))
        for frame in frames:
            frame = frame*255
            frame = frame.astype(np.uint8)
            out.write(frame)
        out.release()
    
    # plot all the frames as an animation (video)
    fig = plt.figure()
    for i in range(len(converted_video_frames)):
        frames.append([plt.imshow(converted_video_frames[i], animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=33.33, blit=True,
                                    repeat_delay=1000)
    plt.axis('off')
    ani.save('mosaic_dogvid.mp4')
    plt.show()
    
if __name__ == '__main__':
    #train()
    test()
    #convert('videos/dogvid.mp4')
