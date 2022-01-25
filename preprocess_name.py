import csv
import numpy as np
import pickle
import random

def make_onehot(dims, i):
    vector = np.zeros(dims)
    vector[i] = 1.0
    return vector

def main():
    csv_file = './data/username/username.csv'
    name_file = './data/username/name.pkl'

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

if __name__ == '__main__':
    main()
