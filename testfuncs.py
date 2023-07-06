from netclasses import VAE
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle
from processpics import crop_size


# Test the syntax and dimensionality of the model on one sample. Expect noise as output
def test_noise_generation(nodes = 64, gray = False):
    model = VAE(10, nodes)
    i = np.random.choice(13234)
    image_path = f'./lfwdatacolor/Faces/image_{i}.jpg'
    if gray:
        image_path = f'./lfwdatagray/Faces/image_{i}.jpg'
    imgnp = cv2.imread(image_path)
    imgnp = cv2.cvtColor(imgnp, cv2.COLOR_BGR2RGB)
    img = torch.reshape(torch.tensor(imgnp, dtype = torch.float32), (1, -1, crop_size[0], crop_size[1]))
    out = model(img)
    plt.imshow(imgnp)
    plt.matshow(out.squeeze().detach().reshape((crop_size[0], crop_size[1], -1)))


def compare_outputs(model, pretrain = True, gray = False):
    image_path = None
    if pretrain:
        i = np.random.choice(13234)
        image_path = f'./lfwdatacolor/Faces/image_{i}.jpg'
        if gray:
            image_path = f'./lfwdatagray/Faces/image_{i}.jpg'
    else:
        i = np.random.choice(97)
        image_path = f'./profdatacolor/Faces/image_{i}.jpg'
        if gray:
            image_path = f'./profdatagray/Faces/image_{i}.jpg'
    imgnp = cv2.imread(image_path)
    imgnp = cv2.cvtColor(imgnp, cv2.COLOR_BGR2RGB)
    img = torch.reshape(torch.tensor(imgnp, dtype = torch.float32), (1, -1, crop_size[0], crop_size[1]))
    out = model(img)
    plt.imshow(imgnp)
    plt.matshow(out.squeeze().detach().reshape((crop_size[0], crop_size[1], -1)))


def compare_and_gen(model):
    compare_outputs(model, pretrain = False)
    model.generate_face()