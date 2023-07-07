import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from processpics import crop_size, image_size

class VAEConv(nn.Module):
    def __init__(self, latent_dims, hidden_nodes):
        super().__init__()
        # Encoding
        self.encode1  = nn.Conv2d(in_channels = 3, out_channels = 4, kernel_size = (32,32))
        self.encode2 = nn.MaxPool2d(kernel_size = (2,2), stride = 1)
        self.encode3  = nn.Linear(876096, hidden_nodes)

        # Variational Part                  
        self.encode_mu = torch.nn.Linear(hidden_nodes, latent_dims)
        self.encode_sigma = torch.nn.Linear(hidden_nodes, latent_dims)

        # Decoding
        self.decode1 = torch.nn.Linear(latent_dims, hidden_nodes)
        self.decode2 = torch.nn.Linear(hidden_nodes, image_size)

        # Other data
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
        self.dims = latent_dims

    def forward(self, x):
        relu = torch.nn.ReLU()
        sigmoid = torch.nn.Sigmoid()
        # Encoding
        x = self.encode1(x)
        x = relu(self.encode2(x))
        x = torch.flatten(x, start_dim = 1, end_dim = -1)
        x = relu(self.encode3(x))
        # Variational part of VAE
        mu = self.encode_mu(x)
        sigma_squared = torch.pow(self.encode_sigma(x), 2)
        z = mu + sigma_squared * self.N.sample(mu.shape)
        # self.kl = (sigma**2 + mu**2 - torch.log(torch.abs(sigma) + 1e-10) - 1/2).sum() # old formulation
        self.kl = 0.5 * (torch.pow(mu, 2) + sigma_squared - torch.log(sigma_squared) - 1).sum()
        nankl = torch.isnan(self.kl).item()
        assert(not nankl)

        #Decoding
        z = self.decode1(z)
        z = relu(z)
        z = self.decode2(z)
        x_hat = sigmoid(z)
        return torch.reshape(x_hat, (-1, 3, crop_size[0], crop_size[1]))
    
    def generate_face(self):
        mu = torch.rand(self.dims)
        sigma = torch.rand(self.dims)
        z = mu + sigma*self.N.sample(mu.shape)
        z = self.decode1(z)
        z = torch.nn.ReLU()(z)
        z = self.decode2(z)
        x_hat = torch.nn.Sigmoid()(z)
        plt.matshow(x_hat.squeeze().detach().reshape((crop_size[0], crop_size[1], -1)))
        return x_hat
    
class ConvTVAE(nn.Module):
    def __init__(self, latent_dims, hidden_nodes):
        super().__init__()

        prehidden = 1600

        # Encoding
        self.conv1  = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = (32,32), stride = 8, padding = 2)
        self.pool1 = nn.MaxPool2d(kernel_size = (4,4), stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (7,7), stride = 2)
        self.pool2 = nn.MaxPool2d(kernel_size = (3,3), stride = 1)
        self.downtohidden = nn.Linear(prehidden, hidden_nodes)

        # Variational Part                  
        self.encode_mu = torch.nn.Linear(hidden_nodes, latent_dims)
        self.encode_sigma = torch.nn.Linear(hidden_nodes, latent_dims)

        # Decoding
        self.uptohidden = torch.nn.Linear(latent_dims, hidden_nodes)
        self.prereshape = torch.nn.Linear(hidden_nodes, prehidden)
        self.convt1 = nn.ConvTranspose2d(in_channels = 16, out_channels = 8, kernel_size = (5,5), stride = 3)
        self.convt2 = nn.ConvTranspose2d(in_channels = 8, out_channels = 4, kernel_size = (8,8), stride = 4)
        self.convt3 = nn.ConvTranspose2d(in_channels = 4, out_channels = 3, kernel_size = (10,10), stride = 4, padding = 17)

        # Other data
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
        self.dims = latent_dims

    def forward(self, x):
        relu = torch.nn.ReLU()
        sigmoid = torch.nn.Sigmoid()
        # Encoding
        x = self.conv1(x)
        x = relu(self.pool1(x))
        x = self.conv2(x)
        beforereshape = relu(self.pool2(x))
        x = torch.flatten(beforereshape, start_dim = 1, end_dim = -1)
        x = relu(self.downtohidden(x))

        # Variational part of VAE
        mu = self.encode_mu(x)
        sigma_squared = torch.pow(self.encode_sigma(x), 2)
        z = mu + sigma_squared * self.N.sample(mu.shape)
        self.kl = 0.5 * (torch.pow(mu, 2) + sigma_squared - torch.log(sigma_squared) - 1).sum()
        assert(not torch.isnan(self.kl).item()) # No items are NaN

        #Decoding
        x_hat = relu(self.uptohidden(z))
        x_hat = relu(self.prereshape(x_hat))
        x_hat = torch.reshape(x_hat, beforereshape.shape)

        # Transposed Convolution
        x_hat = relu(self.convt1(x_hat))
        x_hat = relu(self.convt2(x_hat))
        x_hat = sigmoid(self.convt3(x_hat))
        return x_hat