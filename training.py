import torch

# Doing in-place allows easier saving via pickle
def train_vae(model, dataloader, nepochs=100, inter_epoch  = True, regularize = False):
    model.train()
    optimizer = None
    if regularize:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-2)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(nepochs):
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
            
            if inter_epoch and batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(dataloader.dataset)))
                
            