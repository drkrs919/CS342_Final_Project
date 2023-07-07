import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from processpics import crop_size, image_size

class VAE(nn.Module):
    def __init__(self, latent_dims, hidden_nodes):
        super().__init__()
        self.encode1 = torch.nn.Linear(in_features = image_size, out_features = hidden_nodes)
        self.encode_mu = torch.nn.Linear(hidden_nodes, latent_dims)
        self.encode_sigma = torch.nn.Linear(hidden_nodes, latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
        self.dims = latent_dims

        self.decode1 = torch.nn.Linear(latent_dims, hidden_nodes)
        self.decode2 = torch.nn.Linear(hidden_nodes, image_size)

    def forward(self, x):
        relu = torch.nn.ReLU()
        sigmoid = torch.nn.Sigmoid()
        # Encoding
        x = torch.flatten(x, start_dim = 1)
        x = self.encode1(x)
        x = relu(x)
        # Variational part of VAE
        mu = self.encode_mu(x)
        sigma = torch.exp(self.encode_sigma(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(torch.abs(sigma) + 1e-10) - 1/2).sum()

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

class VAEDropout(nn.Module):
    def __init__(self, latent_dims, hidden_nodes):
        super().__init__()
        self.encode1 = torch.nn.Linear(in_features = image_size, out_features = hidden_nodes)
        self.encode_mu = torch.nn.Linear(hidden_nodes, latent_dims)
        self.encode_sigma = torch.nn.Linear(hidden_nodes, latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
        self.dims = latent_dims

        self.decode1 = torch.nn.Linear(latent_dims, hidden_nodes)
        self.decode2 = torch.nn.Linear(hidden_nodes, image_size)

    def forward(self, x):
        relu = torch.nn.ReLU()
        sigmoid = torch.nn.Sigmoid()
        # Encoding
        x = torch.flatten(x, start_dim = 1)
        x = self.encode1(x)
        x = torch.nn.Dropout(p = 0.5)(x)
        x = relu(x)
        # Variational part of VAE
        mu = self.encode_mu(x)
        sigma = torch.exp(self.encode_sigma(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(torch.abs(sigma) + 1e-10) - 1/2).sum()

        #Decoding
        z = self.decode1(z)
        z = torch.nn.Dropout(p = 0.5)(z)
        z = relu(z)
        z = self.decode2(z)
        z = torch.nn.Dropout(p = 0.5)(z)
        x_hat = sigmoid(z)
        return torch.reshape(x_hat, (-1, 3, crop_size[0], crop_size[1]))
    
    def generate_face(self):
        self.eval()
        mu = torch.rand(self.dims)
        sigma = torch.rand(self.dims)
        z = mu + sigma*self.N.sample(mu.shape)
        z = self.decode1(z)
        z = torch.nn.ReLU()(z)
        z = self.decode2(z)
        x_hat = torch.nn.Sigmoid()(z)
        plt.matshow(x_hat.squeeze().detach().reshape((crop_size[0], crop_size[1], -1)))
        return x_hat
    