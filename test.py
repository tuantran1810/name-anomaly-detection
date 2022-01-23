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
lstm_hidden_dims1 = 64
lstm_hidden_dims2 = 32
latent_dims = 16
device = 'cpu'

def main():
    all_names = None
    keymap = None
    with open('./name.pkl', 'rb') as fd:
        all_names = pickle.load(fd)
    with open('./keymap.pkl', 'rb') as fd:
        keymap = pickle.load(fd)

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
    vae.load(folder='./model/3')
    vae.eval()

    results = list()
    all_strings = list()
    for s, x in tqdm(dataloader):
        with torch.no_grad():
            out, _, _, _ = vae(x)
            loss = nn.functional.l1_loss(out, x, reduction='none')
            loss = loss.detach().cpu().numpy()
            loss = np.sum(loss, axis=(1,2))
            results.append(loss)
            all_strings.extend(s)
    results = np.concatenate(results, axis=0)

    items = list()
    for i in range(len(results)):
        result = results[i]
        string = all_strings[i]
        items.append({'name': string, 'loss': result})
    items = sorted(items, key=lambda x: x['loss'], reverse=True)

    with open('./test_result.csv', 'w', newline='') as fd:
        fields = ['name', 'loss']
        writer = csv.DictWriter(fd, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow(item)




    # idx = np.where(results > np.percentile(results, 99))
    # output = list()
    # for i in idx[0]:
    #     item = (all_strings[i], results[i])
    #     output.append(item)

    # output = sorted(output, key = lambda x: x[1], reverse=True)
    # print('\n'.join((f"{x[0]} === {x[1]}" for x in output)))

if __name__ == '__main__':
    main()
