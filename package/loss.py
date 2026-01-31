import torch
from torch import nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mse_loss(h, x):
    mse = nn.MSELoss()
    loss = mse(h, x)
    return loss