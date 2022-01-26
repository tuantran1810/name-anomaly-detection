import csv
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from vae import VariationalCharacterAutoEncoder, vae_loss
from vocab import PTB
from tqdm import tqdm
from matplotlib import pyplot as plt

sequence_length = 30
embedding_size = 64
hidden_size = 64
latent_size = 32
embedding_dropout = 0.5
batchsize = 256
device='cpu'

def main():
    ptb = PTB(
        part='test',
        vocab_file='./data/useradd/vocab.pkl',
        data_dir='./data/useradd/token.pkl',
        max_sequence_length=sequence_length,
    )

    params = {
        'batch_size': 256,
        'shuffle': False,
        'num_workers': 2,
        'drop_last': False,
    }
    dataloader = DataLoader(ptb, **params)

    vae = VariationalCharacterAutoEncoder(
        [ptb.sos_idx, ptb.eos_idx, ptb.pad_idx, ptb.unk_idx],
        vocab_size=ptb.vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        embedding_dropout=embedding_dropout,
        latent_size=latent_size,
        max_sequence_length=sequence_length,
        device=device
    )
    vae.load(folder='./model/useradd/44')
    vae.eval()

    trust_lst = list()
    all_strings = list()
    for string, x, y, length in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        length = length.to(device)
        with torch.no_grad():
            logp, mean, logv, z = vae(x, length)
            loss, _ = vae_loss(logp, y, length, mean, logv, reduction='none')
            loss = loss.detach().cpu().numpy()
            trust = -np.log(loss + 10e-12)
            trust_lst.append(trust)
            all_strings.extend(string)
    trust_lst = np.concatenate(trust_lst, axis=0)

    items = list()
    for i in range(len(trust_lst)):

        items.append({
            'name': all_strings[i],
            'trust': trust_lst[i],
        })
    items = sorted(items, key=lambda x: x['trust'])

    with open('./test_result.csv', 'w', newline='') as fd:
        fields = ['name', 'trust']
        writer = csv.DictWriter(fd, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow(item)

    plt.figure()
    plt.hist(sorted(trust_lst)[:7500], bins=200)
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
