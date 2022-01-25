import os
import io
import json
import pickle
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

from utils import OrderedCounter

class PTB(Dataset):
    def __init__(
        self,
        part='train',
        raw_data_dir='./data/name.pkl',
        vocab_file='./data/vocab.pkl',
        data_dir='./data/token.pkl',
        max_sequence_length=30,
        min_ooc=50,
    ):
        super().__init__()
        self.raw_data_dir = raw_data_dir
        self.data_dir = data_dir
        self.part = part
        self.max_sequence_length = max_sequence_length
        self.min_occ = min_ooc
        self.vocab_file = vocab_file

        if not os.path.exists(data_dir) or not os.path.exists(vocab_file):
            print("create new data and vocab")
            self._create_data()
        else:
            self._load_data()

    def __len__(self):
        return len(self.data[self.part])

    def __getitem__(self, idx):
        return (
            self.data[self.part][idx]['string'],
            np.asarray(self.data[self.part][idx]['input']),
            np.asarray(self.data[self.part][idx]['target']),
            self.data[self.part][idx]['length'],
        )

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):
        with open(self.data_dir, 'rb') as fd:
            self.data = pickle.load(fd)
        if vocab:
            self._load_vocab()

    def _load_vocab(self):
        with open(self.vocab_file, 'rb') as fd:
            vocab = pickle.load(fd)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def __create_data_item(self, name):
        chars = [c for c in name]
        inp = ['<sos>'] + chars
        inp = inp[:self.max_sequence_length]

        target = chars[:self.max_sequence_length-1]
        target = target + ['<eos>']

        assert len(inp) == len(target), "%i, %i"%(len(inp), len(target))
        length = len(inp)

        inp.extend(['<pad>'] * (self.max_sequence_length-length))
        target.extend(['<pad>'] * (self.max_sequence_length-length))

        inp = [self.w2i.get(w, self.w2i['<unk>']) for w in inp]
        target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

        item = {
            'string': name,
            'input': inp,
            'target': target,
            'length': length,
        }
        return item

    def _create_data(self):
        self._create_vocab()
        data = dict()
        with open(self.raw_data_dir, 'rb') as fd:
            all_names = pickle.load(fd)
            for part in ['train', 'val', 'test']:
                data[part] = [self.__create_data_item(name) for name in all_names[part]]
        with open(self.data_dir, 'wb') as fd:
            pickle.dump(data, fd)

        self._load_data(vocab=False)

    def _create_vocab(self):
        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with open(self.raw_data_dir, 'rb') as fd:
            all_names = pickle.load(fd)
            all_names = all_names['train']
            for name in all_names:
                for c in name:
                    w2c.update(c)

            for w, c in w2c.items():
                if c > self.min_occ and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with open(self.vocab_file, 'wb') as fd:
            pickle.dump(vocab, fd)

        self._load_vocab()

if __name__ == '__main__':
    # ptb = PTB(
    #     part='train',
    #     raw_data_dir='./data/username/name.pkl',
    #     vocab_file='./data/username/vocab.pkl',
    #     data_dir='./data/username/token.pkl',
    #     max_sequence_length=30,
    #     min_ooc=50,
    # )

    # ptb = PTB(
    #     part='train',
    #     raw_data_dir='./data/useradd/add.pkl',
    #     vocab_file='./data/useradd/vocab.pkl',
    #     data_dir='./data/useradd/token.pkl',
    #     max_sequence_length=30,
    #     min_ooc=50,
    # )
    pass
