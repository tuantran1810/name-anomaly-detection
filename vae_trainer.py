import sys, os
sys.path.append(os.path.dirname(__file__))
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from vae import LSTMVariationalAutoencoder, vae_loss
from utils import ArrayDataset, TensorMaker
from tensorboardX import SummaryWriter

class VAETrainer():
    def __init__(self,
        strings_pkl_path='./name.pkl',
        keymap_pkl_path='./keymap.pkl',
        output_model_path='./model',
        sequence_length = 30,
        feature_dims=100,
        lstm_hidden_dims1=128,
        lstm_hidden_dims2=64,
        latent_dims=32,
        batchsize=512,
        lr=10e-4,
        device='cpu',
    ):
        all_strings = None
        keymap = None
        with open(strings_pkl_path, 'rb') as fd:
            all_strings = pickle.load(fd)
        with open(keymap_pkl_path, 'rb') as fd:
            keymap = pickle.load(fd)
        
        train_strings = all_strings['train']
        val_strings = all_strings['val']

        tensor_maker = TensorMaker(keymap, sequence_length)
        train_dataset = ArrayDataset(train_strings, tensor_maker.make)
        val_dataset = ArrayDataset(val_strings, tensor_maker.make)
        params = {
            'batch_size': batchsize,
            'shuffle': True,
            'num_workers': 2,
            'drop_last': False,
        }
        self.__train_dataloader = DataLoader(train_dataset, **params)
        self.__val_dataloader = DataLoader(val_dataset, **params)


        self.__vae = LSTMVariationalAutoencoder(
            sequence_length=sequence_length,
            feature_dims=feature_dims,
            lstm_hidden_dims1=lstm_hidden_dims1,
            lstm_hidden_dims2=lstm_hidden_dims2,
            latent_dims=latent_dims,
            device=device,
        )
        self.__optim = torch.optim.Adam(self.__vae.parameters(), lr=lr)

        self.__output_model_path = output_model_path
        self.__device = device
        self.__writer = SummaryWriter(log_dir='log')

    def train(self, epochs=10):
        print(f"start training, {epochs} epochs")
        total_samples = len(self.__train_dataloader)
        for epoch in range(epochs):
            epoc_point_offset = total_samples*epoch
            print(f"running on epoch {epoch}")
            self.__vae.train()
            for i, (_, x) in enumerate(tqdm(self.__train_dataloader)):
                ipoint = epoc_point_offset+i
                x = x.to(self.__device)
                self.__optim.zero_grad()
                yhat, z, z_mean, z_log_var = self.__vae(x)
                loss, bce, kl = vae_loss(x, yhat, z_mean, z_log_var)
                loss.backward()
                self.__optim.step()
                self.__writer.add_scalar('training_loss/total', loss, ipoint)
                self.__writer.add_scalar('training_loss/bce', bce, ipoint)
                self.__writer.add_scalar('training_loss/kl', kl, ipoint)

            self.__vae.eval()
            val_loss = list()
            for i, (_, x) in enumerate(tqdm(self.__val_dataloader)):
                x = x.to(self.__device)
                with torch.no_grad():
                    yhat, z, z_mean, z_log_var = self.__vae(x)
                    loss, _, _ = vae_loss(x, yhat, z_mean, z_log_var)
                    val_loss.append(loss)
            self.__writer.add_scalar('validation_loss', sum(val_loss)/len(val_loss), epoch)
            self.__vae.save(folder=os.path.join(self.__output_model_path, str(epoch)))

        print("done, save the model")
        self.__vae.save(folder=self.__output_model_path)

if __name__ == '__main__':
    trainer = VAETrainer()
    trainer.train(10)
