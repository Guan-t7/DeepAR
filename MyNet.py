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
BATCH_SIZE = 256
N_EPOCHS = 6

COND_STEPS = 7*24*4
PRED_STEPS = 24*4

N_CLNT = 370

# data
# one sample: a window that forecast [7d:8d), conditioned by [0:7d)
class SeriesDataset(Dataset):
    def __init__(self, z, x, train: bool = True):
        '''from np.ndarray
        z[tm, clnt]
        x[tm, tm_covari]
        '''
        self.z = z
        self.x = x
        self.wnd = COND_STEPS+PRED_STEPS
        # 1D separation between samples
        self.stride = PRED_STEPS

        global N_CLNT
        N_CLNT = self.z.shape[1]
        # actual wnd sz go beyond for 1 dp due to lstm char
        self.ser_len = (self.z.shape[0] - self.wnd - 1) // self.stride + 1

    def __len__(self):
        return N_CLNT * self.ser_len

    def __getitem__(self, index):
        data_i = index // self.ser_len
        index -= self.ser_len * data_i
        index *= self.stride
        # past observations; scale handling
        z = self.z[index: index + self.wnd, data_i]
        v = z[:COND_STEPS].mean(keepdims=True) + 1
        # covariates
        x = self.x[index + 1: index + 1 + self.wnd]
        data = np.concatenate((np.expand_dims(z, 1), x), axis=1)
        # forecast
        index += COND_STEPS
        label = self.z[index + 1: index + 1 + PRED_STEPS, data_i]
        return data_i, data, v, label

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
        
        self.embedding = nn.Embedding(N_CLNT, 20)
        self.lstm = nn.LSTM(input_size=1+5+20, hidden_size=40, num_layers=3,
                            batch_first=True)
        self.distribution_mu = nn.Linear(40, 1)
        self.distribution_sigma = nn.Sequential(
            nn.Linear(40, 1),
            nn.Softplus(),
        )

    def forward(self, idx: 'torch.IntTensor', x: 'torch.Tensor', hx=None):
        '''
        x[batch, seq_len, feat]
        idx[batch]
        '''
        embed = self.embedding(idx)
        # [batch, feat] -> [batch, seq_len, feat]
        embed = torch.unsqueeze(embed, 1).expand(-1, x.shape[1], -1)
        lstm_in = torch.cat((x, embed), 2)
        lstm_out, hx = self.lstm(lstm_in, hx)
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

# training loop
def fit(epochs, model, loss_fn, opt, train_dl, valid_dl):
    Nb_batch = 0
    loss_train = .0
    for epoch in range(epochs):
        model.train()
        for data_i, data, v, label in train_dl:
            data_i, data, = data_i.to(DEVICE), data.to(DEVICE),
            v, label = v.to(DEVICE), label.to(DEVICE)
            # scale input z
            data[...,0] /= v
            # feed nn a mini-batch. Forward pass
            mu, sigma, hx = model(data_i, data)
            # scale output
            mu = mu[:, -PRED_STEPS:] * v
            sigma = sigma[:, -PRED_STEPS:] * v
            loss = loss_fn(mu, sigma, label)
            # Backward pass
            if opt:
                loss.backward()
                opt.step()
                opt.zero_grad()
            # accumulate loss
            loss_train += loss.item()
            LOG_EVERY = 4  # ... mini-batches
            if Nb_batch % LOG_EVERY == 0:
                writer.add_scalar('Loss/train', loss_train / LOG_EVERY,
                                Nb_batch)
                loss_train = .0
            Nb_batch += 1
        model.eval()

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
