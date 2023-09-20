import torch
import numpy as np
from tqdm.notebook import tqdm

# Doing in-place allows easier saving via pickle
def train_vae(model, dataloader, nepochs=100, lr = 1e-3, scale_KL  = True, regularize = False):
    model.train()
    optimizer = None
    if regularize:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-2)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    try:
        progress = tqdm(range(nepochs), position = 0, unit = "epoch")
        shortepoch = len(dataloader) <= 10
        for epoch in progress: # Has a counter that increments each epoch
            train_loss = 0
            batches = dataloader
            if not shortepoch:
                batches = tqdm(dataloader, position = 1, unit = "batch", leave = False)
            for batch_idx, (data, _) in enumerate(batches):
                optimizer.zero_grad()
                x_hat = model(data)
                term1 = (np.square(data - x_hat)).mean()
                term2 = model.kl * 1e-5
                scaling = -1
                if scale_KL:
                    percentdone = batch_idx/len(dataloader)
                    scaling = 0.1
                    term2 = model.kl * lr * np.square(percentdone) * scaling
                loss = term1 + term2
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                if not shortepoch:
                    batches.set_description(f"Batch [{batch_idx+1}/{len(dataloader)}]")
                    batches.set_postfix(KL_Scaled = term2, KL_Raw = model.kl.item(), Reconstruction_Loss = term1.item())
            progress.set_description(f"Epoch [{epoch+1}/{nepochs}]")
            progress.set_postfix(KL_Raw = model.kl.item(), Reconstruction_Loss = term1.item())
    except KeyboardInterrupt:
        print('Exited from training early')
        
