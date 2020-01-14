import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from glove_dataset import GloveDataset
from glove_model import GloveModel

EMBED_DIM = 300

N_EPOCHS = 100
BATCH_SIZE = 2048
X_MAX = 100
ALPHA = 0.75

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def weight_func(x, x_max, alpha):
    wx = (x / x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    return wx


def wmse_loss(weights, inputs, targets):
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    return torch.mean(loss)


dataset = GloveDataset(open("./text8").read(), n_words=10000000)
glove = GloveModel(dataset._vocab_len, EMBED_DIM)
glove.to(device)


n_batches = int(len(dataset._xij) / BATCH_SIZE)
loss_values = list()

optimizer = optim.Adagrad(glove.parameters(), lr=0.05)

if __name__ == "__main__":
    for e in range(1, N_EPOCHS + 1):
        batch_i = 0

        for x_ij, i_idx, j_idx in dataset.get_batches(BATCH_SIZE):

            batch_i += 1
            optimizer.zero_grad()

            outputs = glove(i_idx.to(device), j_idx.to(device))
            weights_x = weight_func(x_ij, X_MAX, ALPHA)
            loss = wmse_loss(weights_x, outputs, torch.log(x_ij))
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())
            if batch_i % 250 == 0:
                print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(
                    e, N_EPOCHS, batch_i, n_batches, np.mean(loss_values[-20:])))

        print("Saving model...")
        torch.save(glove.state_dict(), "text8.pt")