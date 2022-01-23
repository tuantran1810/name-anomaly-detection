import pickle
import random
import csv
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from utils import ArrayDataset, TensorMaker
from vae import LSTMVariationalAutoencoder
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from tqdm import tqdm

sequence_length = 30
feature_dims = 100
lstm_hidden_dims1 = 128
lstm_hidden_dims2 = 64
latent_dims = 32
device = 'cpu'

def tensor_to_string(keyidx, tensor):
    out = list()
    idx = np.argmax(tensor, axis=0)
    for i in idx:
        c = keyidx[i]
        out.append(c)
    return ''.join(out)

def main():
    all_names = None
    keymap = None
    keyidx = None
    with open('./name.pkl', 'rb') as fd:
        all_names = pickle.load(fd)
    with open('./keymap.pkl', 'rb') as fd:
        keymap = pickle.load(fd)
    with open('./keyidx.pkl', 'rb') as fd:
        keyidx = pickle.load(fd)

    all_names = all_names['test']
    random.shuffle(all_names)
    tensor_maker = TensorMaker(keymap, sequence_length)
    dataset = ArrayDataset(all_names, tensor_maker.make)
    params = {
        'batch_size': 512,
        'shuffle': False,
        'num_workers': 2,
        'drop_last': False,
    }
    dataloader = DataLoader(dataset, **params)

    vae = LSTMVariationalAutoencoder(
        sequence_length=sequence_length,
        feature_dims=feature_dims,
        lstm_hidden_dims1=lstm_hidden_dims1,
        lstm_hidden_dims2=lstm_hidden_dims2,
        latent_dims=latent_dims,
        device='cpu',
    )
    vae.load(folder='./model/celoss')
    vae.eval()

    string = "tuấn nè"
    string, tensor = tensor_maker.make(string)
    tensor = torch.unsqueeze(tensor, dim=0)
    with torch.no_grad():
        out, _, _, _ = vae(tensor)
    out = out.detach().cpu().numpy()
    print(tensor_to_string(keyidx, out[0]))

if __name__ == '__main__':
    main()
