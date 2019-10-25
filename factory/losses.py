import torch.nn as nn


def mse():
    return nn.MSELoss()


def get_loss(loss_name):
    print('loss name:', loss_name)
    f = globals().get(loss_name)
    return f()
