import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# train process logging
today = datetime.datetime.today()
writer = SummaryWriter(purge_step=0, )

# check available device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
BATCH_SIZE = 64
# N_EPOCHS = 20
N_EPOCHS = 8

# data
# one sample: a window that forecast [7d:8d), conditioned by [0:7d)
class SeriesDataset(Dataset):
    def __init__(self, z, x, train: bool = True):
        self.z = z
        self.x = x

    def __len__(self):
        return self.z.shape[0] - 8*24*4 - 1

    def __getitem__(self, index):
        # todo 1D sep betw samples
        z = self.z[index:index + 8*24*4]
        x = self.x[index + 1:index + 1 + 8*24*4]
        label = self.z[index + 1:index + 8*24*4 + 1]
        data = np.concatenate((np.expand_dims(z, 1), x), axis=1)
        return data, label

def get_data(bs):
    train_z = np.load(f"data/train_z.npy").astype('float32')
    test_z = np.load(f"data/test_z.npy",).astype('float32')
    train_x = np.load(f"data/train_x.npy").astype('float32')
    test_x = np.load(f"data/test_x.npy",).astype('float32')
    train_ds = SeriesDataset(train_z, train_x)
    test_ds = SeriesDataset(test_z, test_x, train=False)
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(test_ds, batch_size=bs * 2),
    )

# model
class MyNet(nn.Module):
    def __init__(self, params=None):
        super(MyNet, self).__init__()
        self.lstm = nn.LSTM(input_size=1+5, hidden_size=40, num_layers=3, batch_first=True)
        self.distribution_mu = nn.Linear(40, 1)
        self.distribution_sigma = nn.Sequential(
            nn.Linear(40, 1),
            nn.Softplus(),
        ) 

    def forward(self, x: 'torch.Tensor[float, float, float]', hx=None):
        lstm_out, hx = self.lstm(x, hx)
        mu = self.distribution_mu(lstm_out)
        sigma = self.distribution_sigma(lstm_out)
        return mu.squeeze(), sigma.squeeze(), hx

def get_model():
    model = MyNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),)

    # maximizing the log-likelihood; log of prob_density
    def loss_fn(mu, sigma, labels: 'torch.Tensor[int]'):
        distrib = torch.distributions.normal.Normal(mu, sigma)
        log_likelihood = distrib.log_prob(labels)
        return -torch.mean(log_likelihood)

    return model, optimizer, loss_fn

# feed nn a mini-batch, returning the loss
def loss_batch(model, loss_fn, data, label, opt=None):
    hx = loss_batch.hx
    # Forward pass
    mu, sigma, hx = model(data, hx)
    loss_batch.hx = hx
    loss = loss_fn(mu, sigma, label)

    # Backward pass
    if opt:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item()

# training loop
def fit(epochs, model, loss_fn, opt, train_dl, valid_dl):
    Nb_batch = 0
    loss_train = .0
    for epoch in range(epochs):
        model.train()
        for data, label in train_dl:
            loss_batch.hx = None
            # |condition|forecast|; treated the same in train
            data, label = data.to(DEVICE), label.to(DEVICE)
            loss_train += loss_batch(model, loss_fn, data, label, opt)
            # log every ... mini-batches
            if Nb_batch % 32 == 0:
                writer.add_scalar('Loss/train', loss_train / (32*BATCH_SIZE),
                                Nb_batch)
                loss_train = .0
            Nb_batch += 1
        model.eval()
    #     with torch.no_grad():
    #         running_loss = 0.0
    #         for i, data in enumerate(valid_dl):
    #             xb, yb = data
    #             xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    #             loss = loss_batch(model, loss_fn, xb, yb)
    #             running_loss += loss

    # # log a Matplotlib Figure showing the model's predictions on a
    # # random mini-batch
    # images, labels = next(iter(valid_dl))
    # images, labels = images.to(DEVICE), labels.to(DEVICE)
    # with torch.no_grad():
    #     writer.add_figure('predictions vs. actuals',
    #                       plot_classes_preds(model, images, labels))
    return model, opt


def main():
    # prep data and model
    train_dl, test_dl = get_data(BATCH_SIZE)
    model, opt, loss_fn = get_model()
    # train
    model, opt = fit(N_EPOCHS, model, loss_fn, opt, train_dl, test_dl)
    # save model
    torch.save(model, f"MyNet_{today.day}_{today.hour}-{today.minute}.pt")


if __name__ == '__main__':
    main()
