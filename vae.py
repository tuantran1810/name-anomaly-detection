import sys, os
sys.path.append(os.path.dirname(__file__))
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils

def vae_loss(logp, target, length, mean, logv, reduction='mean'):
    # cut-off unnecessary padding from target, and flatten
    max_length = torch.max(length).item()
    target = target[:, :max_length].contiguous().view(-1)
    logp = logp.view(-1, logp.size(2))

    # Negative Log Likelihood
    nll_loss = nn.functional.nll_loss(logp, target, reduction=reduction)
    if reduction == 'none':
        nll_loss = nll_loss.reshape(-1, max_length)
        nll_loss = torch.mean(nll_loss, dim=1)

    # KL Divergence
    kl_loss = None
    if reduction == 'none':
        kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp(), dim=1)
    else:
        kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

    return nll_loss, kl_loss

def vae_anneal_loss(logp, target, length, mean, logv, step, k=0.0025, x0=2500):
    nll_loss, kl_loss = vae_loss(logp, target, length, mean, logv)
    kl_weight = float(1/(1+np.exp(-k*(step-x0))))
    return nll_loss, kl_loss, kl_weight

class VariationalCharacterAutoEncoder(nn.Module):
    def __init__(
        self,
        special_token_idx, # [sos_idx, eos_idx, pad_idx, unk_idx]
        vocab_size=142,
        embedding_size=64,
        hidden_size=32,
        word_dropout=0,
        embedding_dropout=0.5,
        latent_size=16,
        max_sequence_length=30,
        num_layers=1,
        bidirectional=True,
        device='cpu',
    ):
        super(VariationalCharacterAutoEncoder, self).__init__()
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.sos_idx, self.eos_idx, self.pad_idx, self.unk_idx = special_token_idx

        self.latent_size = latent_size

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.encoder_rnn = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.decoder_rnn = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)
        
        self.to(device)

    def forward(self, input_sequence, length):
        input_sequence = input_sequence.to(self.device)
        length = length.to(self.device)

        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        _, (hidden, _) = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = torch.permute(hidden, (1,0,2))
            hidden = hidden.reshape(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)
        z = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = z * std + mean

        # # DECODER
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.reshape(batch_size, self.hidden_factor, self.hidden_size)
            hidden = torch.permute(hidden, (1,0,2))
        else:
            hidden = hidden.unsqueeze(0)
        hidden = hidden.contiguous()

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, (hidden, torch.zeros_like(hidden).to(self.device)))

        # # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b, s, _ = padded_outputs.size()

        # # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp, mean, logv, z

    def save(self, folder='./model'):
        Path(folder).mkdir(exist_ok=True, parents=True)
        path = os.path.join(folder, 'model.pth')
        torch.save(self.state_dict(), path)

    def load(self, folder='./model'):
        path = os.path.join(folder, 'model.pth')
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)

if __name__ == '__main__':
    vae = VariationalCharacterAutoEncoder([0,1,2,3])
    x = torch.zeros(5, 30).long()
    length = torch.tensor([x+10 for x in range(5)])
    logp, mean, logv, z = vae(x, length)
    nll_loss, kl_loss, kl_weight = vae_anneal_loss(logp, torch.zeros(5, 30).long(), length, mean, logv, 1000)
    print(nll_loss, kl_loss, kl_weight)
