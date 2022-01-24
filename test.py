import csv
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from vae import VariationalCharacterAutoEncoder, vae_anneal_loss
from vocab import PTB
from tqdm import tqdm

sequence_length = 30
embedding_size = 64
hidden_size = 64
latent_size = 16
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
    vae.load(folder='./model/0')
    vae.eval()

    results = list()
    all_strings = list()
    for string, x, y, length in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        length = length.to(device)
        with torch.no_grad():
            logp, mean, logv, z = vae(x, length)
            max_length = torch.max(length).item()
            y = y[:, :max_length].contiguous().view(-1)
            logp = logp.view(-1, logp.size(2))
            loss = nn.functional.nll_loss(logp, y, reduction='none')
            loss = loss.reshape(-1, max_length)
            loss = torch.mean(loss, dim=1)
            loss = loss.detach().cpu().numpy()
            results.append(loss)
            all_strings.extend(string)
    results = np.concatenate(results, axis=0)

    items = list()
    print(len(results))
    print(len(all_strings))
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

if __name__ == '__main__':
    main()
