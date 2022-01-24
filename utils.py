import sys, os
sys.path.append(os.path.dirname(__file__))
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter, OrderedDict

class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

class ArrayDataset(Dataset):
    def __init__(self, array, post_processing):
        self.__data = array
        self.__post_processing  = post_processing

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, index):
        data = self.__data[index]
        return self.__post_processing(data)

class TensorMaker():
    def __init__(self, keymap, sequence_length):
        self.__keymap = keymap
        self.__sequence_length = sequence_length
        self.__unk = keymap['<unk>']
        self.__pad = keymap['<pad>']
        self.__all_pad = np.tile(self.__pad, (sequence_length, 1))

    def make(self, string):
        length = min(self.__sequence_length, len(string))
        tensor = self.__all_pad[:]
        for i in range(length):
            c = string[i]
            if c in self.__keymap:
                tensor[i] = self.__keymap[c]
            else:
                tensor[i] = self.__unk
        return string, torch.tensor(tensor).float()
