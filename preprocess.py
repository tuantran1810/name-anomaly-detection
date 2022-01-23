import csv
import numpy as np
import pickle
import random

def make_onehot(dims, i):
    vector = np.zeros(dims)
    vector[i] = 1.0
    return vector

def main():
    csv_file = './username.csv'
    name_file = './name.pkl'
    keymap_file = './keymap.pkl'
    keyidx_file = './keyidx.pkl'
    dims = 100

    all_names = list()
    with open(csv_file) as fd:
        reader = csv.DictReader(fd, delimiter=',')
        for row in reader:
            name = row['full_name']
            all_names.append(name.lower())

    random.shuffle(all_names)
    n_train = int(0.9*len(all_names))
    train_names = all_names[:n_train]
    val_names = train_names[:5000]
    train_names = train_names[5000:]
    test_names = all_names[n_train:]
    data = {
        'train': train_names,
        'val': val_names,
        'test': test_names,
    }

    with open(name_file, 'wb') as fd:
        pickle.dump(data, fd)

    all_chars = dict()
    for name in all_names:
        for c in name:
            if c in all_chars:
                all_chars[c] = all_chars[c] + 1
            else:
                all_chars[c] = 1

    all_chars = sorted(all_chars.items(), key = lambda item: item[1], reverse = True)
    n_chars = dims - 2
    popular_chars = [c[0] for c in all_chars[:n_chars]]
    keymap = dict()
    keyidx = dict()

    for i in range(len(popular_chars)):
        c = popular_chars[i]
        c_onehot = make_onehot(dims, i)
        keymap[c] = c_onehot
        keyidx[i] = c

    keymap['<unk>'] = make_onehot(dims, dims-2)
    keymap['<pad>'] = make_onehot(dims, dims-1)
    keyidx[dims-2] = '<unk>'
    keyidx[dims-1] = '<pad>'

    with open(keymap_file, 'wb') as fd:
        pickle.dump(keymap, fd)
    with open(keyidx_file, 'wb') as fd:
        pickle.dump(keyidx, fd)

if __name__ == '__main__':
    main()
