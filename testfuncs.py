from netclasses import VAEConv
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from processpics import crop_size
from timeit import default_timer as timer

def test_model(model = None, pretrain = True, gray = False):
    usemodel = model
    if usemodel == None:
        usemodel = VAEConv(5, 32)
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
    if imgnp is None:
        test_model(usemodel, pretrain = pretrain, gray = gray)
    else:
        imgnp = cv2.cvtColor(imgnp, cv2.COLOR_BGR2RGB)
        img = torch.reshape(torch.tensor(imgnp, dtype = torch.float32), (1, -1, crop_size[0], crop_size[1]))

        # testimg = img.squeeze().detach().reshape((crop_size[0], crop_size[1], 3)).numpy().astype(int)
        # plt.imshow(testimg)

        out = usemodel(img)
        outnp = (out.squeeze().detach().reshape((crop_size[0], crop_size[1], 3)).numpy() * 255).astype(int)
        plt.imshow(imgnp)
        plt.imshow(outnp)
        return outnp


def compare_and_gen(model):
    test_model(model = model, pretrain = False)
    model.generate_face()

def func_runtime(function, name):
    start = timer()
    function
    end = timer()
    time = end - start
    print(f"function {name} took {round(time, 2)}s")