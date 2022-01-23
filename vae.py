import sys, os
sys.path.append(os.path.dirname(__file__))
from genericpath import exists
from pathlib import Path
import torch
from torch import nn

def vae_loss(x, yhat, z_mean, z_log_var):
    '''
    x, yhat:             |    [batchsize, chars, features]
    z_mean, z_log_var:   |    [batchsize, latent_dims]
    '''
    batchsize, chars, features = x.shape
    x = torch.reshape(x, (batchsize, chars*features))
    yhat = torch.reshape(yhat, (batchsize, -1))
    bce = nn.functional.binary_cross_entropy(yhat, x)
    kl_divergence = -0.5 * torch.sum(1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
    return bce + kl_divergence, bce, kl_divergence

class Encoder(nn.Module):
    def __init__(self,
        input_dims=100,
        lstm_hidden_dims1=64,
        lstm_hidden_dims2=32,
        latent_dims=16,
        device='cpu',
    ):
        super(Encoder, self).__init__()
        self.__lstm1 = nn.LSTM(
            input_size=input_dims,
            hidden_size=lstm_hidden_dims1,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        )
        self.__lstm2 = nn.LSTM(
            input_size=lstm_hidden_dims1,
            hidden_size=lstm_hidden_dims2,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        )
        self.__fc_mean = nn.Linear(lstm_hidden_dims2, latent_dims)
        self.__fc_log_var = nn.Linear(lstm_hidden_dims2, latent_dims)

        self.to(device)
        self.__device = device

    def __sampling(self, z_mean, z_log_var):
        epsilon = torch.normal(0, 1, size=(z_mean.shape)).to(self.__device)
        return z_mean + torch.exp(z_log_var)*epsilon

    def forward(self, x):
        '''
        x: [batchsize, chars, features]
        '''
        x = x.to(self.__device)
        x, _ = self.__lstm1(x)
        x = nn.functional.dropout(x, p=0.2)
        _, (x, _) = self.__lstm2(x)
        x = x[1]
        z_mean = self.__fc_mean(x)
        z_log_var = self.__fc_log_var(x)
        z = self.__sampling(z_mean, z_log_var)
        return (z, z_mean, z_log_var)

class Decoder(nn.Module):
    def __init__(self,
        sequence_length = 30,
        latent_dims=16,
        lstm_hidden_dims1=32,
        lstm_hidden_dims2=64,
        output_dims=100,
        device='cpu',
    ):
        super(Decoder, self).__init__()
        self.__lstm1 = nn.LSTM(
            input_size=latent_dims,
            hidden_size=lstm_hidden_dims1,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        )
        self.__lstm2 = nn.LSTM(
            input_size=lstm_hidden_dims1,
            hidden_size=lstm_hidden_dims2,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        )
        self.__fc = nn.Linear(lstm_hidden_dims2, output_dims)

        self.to(device)
        self.__sequence_length = sequence_length
        self.__device = device

    def forward(self, x):
        '''
        x: [batchsize, features]
        '''
        x = x.to(self.__device)
        x = x.unsqueeze(dim=1)
        x = x.repeat(1, self.__sequence_length, 1)
        x, _ = self.__lstm1(x)
        x = nn.functional.dropout(x, p=0.2)
        x, _ = self.__lstm2(x)
        # x = nn.functional.relu(x)
        x = self.__fc(x)
        x = nn.functional.softmax(x, dim=2)
        return x

class LSTMVariationalAutoencoder(nn.Module):
    def __init__(self,
        sequence_length = 30,
        feature_dims=100,
        lstm_hidden_dims1=64,
        lstm_hidden_dims2=32,
        latent_dims=16,
        device='cpu',
    ):
        super(LSTMVariationalAutoencoder, self).__init__()
        self.__encoder = Encoder(
            input_dims=feature_dims,
            lstm_hidden_dims1=lstm_hidden_dims1,
            lstm_hidden_dims2=lstm_hidden_dims2,
            latent_dims=latent_dims,
            device=device,
        )
        self.__decoder = Decoder(
            sequence_length=sequence_length,
            latent_dims=latent_dims,
            lstm_hidden_dims1=lstm_hidden_dims2,
            lstm_hidden_dims2=lstm_hidden_dims1,
            output_dims=feature_dims,
            device=device,
        )
        self.__device = device
        self.to(device)

    def forward(self, x):
        '''
        x: [batchsize, chars, features]
        '''
        x = x.to(self.__device)
        z, z_mean, z_log_var = self.__encoder(x)
        out = self.__decoder(z)
        return out, z, z_mean, z_log_var

    @property
    def encoder(self):
        return self.__encoder

    @property
    def decoder(self):
        return self.__decoder

    def save(self, folder='./model'):
        Path(folder).mkdir(exist_ok=True, parents=True)
        encoder_path = os.path.join(folder, 'encoder.pth')
        decoder_path = os.path.join(folder, 'decoder.pth')
        enc = self.__encoder.to('cpu')
        dec = self.__decoder.to('cpu')
        torch.save(enc.state_dict(), encoder_path)
        torch.save(dec.state_dict(), decoder_path)

    def load(self, folder='./model'):
        encoder_path = os.path.join(folder, 'encoder.pth')
        decoder_path = os.path.join(folder, 'decoder.pth')
        self.__encoder.load_state_dict(torch.load(encoder_path))
        self.__decoder.load_state_dict(torch.load(decoder_path))
        self.to(self.__device)

if __name__ == '__main__':
    vae = LSTMVariationalAutoencoder()
    x = torch.zeros(4, 30, 100)
    out, z, z_mean, z_log_var = vae(x)
    loss = vae_loss(x, out, z_mean, z_log_var)
    vae.save()
