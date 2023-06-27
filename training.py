import torch
from tqdm import tqdm

# Doing in-place allows easier saving via pickle
def train_vae(model, dataloader, nepochs=100, inter_epoch  = True, regularize = False):
    model.train()
    optimizer = None
    if regularize:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-2)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    try:
        progress = tqdm(range(nepochs))
        for epoch in progress: # Has a counter that increments each epoch
            train_loss = 0
            for batch_idx, (data, _) in enumerate(dataloader):
                optimizer.zero_grad()
                x_hat = model(data)
                term1 = ((data - x_hat)**2).mean()
                term2 = model.kl
                loss = term1 + 1e-5*term2  
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            progress.set_description(f"Epoch [{epoch+1}/{nepochs}]")
            progress.set_postfix(Loss = loss.item())
    except KeyboardInterrupt:
        print('Exited from training early')
            