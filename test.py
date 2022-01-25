import csv
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from vae import VariationalCharacterAutoEncoder, vae_loss
from vocab import PTB
from tqdm import tqdm

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
        vocab_file='./data/vocab.pkl',
        data_dir='./data/token.pkl',
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
    vae.load(folder='./model/18')
    vae.eval()

    loss_lst = list()
    all_strings = list()
    for string, x, y, length in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        length = length.to(device)
        with torch.no_grad():
            logp, mean, logv, z = vae(x, length)
            loss, _ = vae_loss(logp, y, length, mean, logv, reduction='none')
            loss_lst.append(loss.detach().cpu().numpy())
            all_strings.extend(string)
    loss_lst = np.concatenate(loss_lst, axis=0)

    items = list()
    for i in range(len(loss_lst)):

        items.append({
            'name': all_strings[i],
            'loss': loss_lst[i],
        })
    items = sorted(items, key=lambda x: x['loss'], reverse=True)

    with open('./test_result.csv', 'w', newline='') as fd:
        fields = ['name', 'loss']
        writer = csv.DictWriter(fd, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow(item)

if __name__ == '__main__':
    main()
