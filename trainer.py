import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from vae import VariationalCharacterAutoEncoder, vae_anneal_loss
from vocab import PTB
from tensorboardX import SummaryWriter

class Trainer():
    def __init__(self,
        token_pkl_path='./data/token.pkl',
        vocab_pkl_path='./data/vocab.pkl',
        output_model_path='./model',
        log_dir='./log',
        sequence_length=30,
        embedding_size=64,
        hidden_size=64,
        latent_size=32,
        embedding_dropout=0.5,
        batchsize=256,
        lr=10e-4,
        device='cpu',
    ):
        ptb_train = PTB(
            part='train',
            vocab_file=vocab_pkl_path,
            data_dir=token_pkl_path,
            max_sequence_length=sequence_length
        )

        ptb_val = PTB(
            part='val',
            vocab_file=vocab_pkl_path,
            data_dir=token_pkl_path,
            max_sequence_length=sequence_length
        )

        params = {
            'batch_size': batchsize,
            'shuffle': True,
            'num_workers': 2,
            'drop_last': False,
        }
        self.__train_dataloader = DataLoader(ptb_train, **params)
        self.__val_dataloader = DataLoader(ptb_val, **params)

        self.__vae = VariationalCharacterAutoEncoder(
            [ptb_train.sos_idx, ptb_train.eos_idx, ptb_train.pad_idx, ptb_train.unk_idx],
            vocab_size=ptb_train.vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            embedding_dropout=embedding_dropout,
            latent_size=latent_size,
            max_sequence_length=sequence_length,
            device=device
        )
        self.__optim = torch.optim.Adam(self.__vae.parameters(), lr=lr)

        self.__output_model_path = output_model_path
        self.__device = device
        self.__writer = SummaryWriter(log_dir=log_dir)

    def train(self, epochs=10):
        print(f"start training, {epochs} epochs")
        total_samples = len(self.__train_dataloader)
        for epoch in range(epochs):
            epoc_point_offset = total_samples*epoch
            print(f"running on epoch {epoch}")

            self.__vae.train()
            for i, (string, x, y, length) in enumerate(tqdm(self.__train_dataloader)):
                ipoint = epoc_point_offset+i

                batchsize = len(string)
                x = x.to(self.__device)
                y = y.to(self.__device)
                length = length.to(self.__device)
                self.__optim.zero_grad()
                logp, mean, logv, z = self.__vae(x, length)
                nll_loss, kl_loss, kl_weight = vae_anneal_loss(logp, y, length, mean, logv, ipoint)
                kl_loss = kl_weight * kl_loss
                loss = nll_loss + kl_loss
                loss.backward()
                self.__optim.step()

                self.__writer.add_scalar('training_loss/nll_loss', nll_loss, ipoint)
                self.__writer.add_scalar('training_loss/kl_loss', kl_loss, ipoint)
                self.__writer.add_scalar('training_loss/kl_weight', kl_weight, ipoint)
                self.__writer.add_scalar('training_loss/loss', loss, ipoint)

            self.__vae.eval()
            val_loss = list()
            for i, (string, x, y, length) in enumerate(tqdm(self.__val_dataloader)):
                x = x.to(self.__device)
                y = y.to(self.__device)
                length = length.to(self.__device)
                with torch.no_grad():
                    logp, mean, logv, z = self.__vae(x, length)
                    nll_loss, kl_loss, kl_weight = vae_anneal_loss(logp, y, length, mean, logv, ipoint)
                    kl_loss = kl_weight * kl_loss
                    loss = nll_loss + kl_loss
                    val_loss.append(loss)
            self.__writer.add_scalar('validation_loss', sum(val_loss)/len(val_loss), epoch)
            self.__vae.save(folder=os.path.join(self.__output_model_path, str(epoch)))

        print("done, save the model")
        self.__vae.save(folder=self.__output_model_path)

if __name__ == '__main__':
    # trainer = Trainer(
    #     token_pkl_path='./data/username/token.pkl',
    #     vocab_pkl_path='./data/username/vocab.pkl',
    #     output_model_path='./model/username',
    #     log_dir='./log',
    #     sequence_length=30,
    #     embedding_size=64,
    #     hidden_size=64,
    #     latent_size=32,
    #     embedding_dropout=0.5,
    #     batchsize=256,
    #     lr=10e-4,
    #     device='cpu',
    # )

    trainer = Trainer(
        token_pkl_path='./data/useradd/token.pkl',
        vocab_pkl_path='./data/useradd/vocab.pkl',
        output_model_path='./model/useradd',
        log_dir='./log',
        sequence_length=30,
        embedding_size=64,
        hidden_size=64,
        latent_size=32,
        embedding_dropout=0.5,
        batchsize=256,
        lr=10e-4,
        device='cpu',
    )
    trainer.train()

    pass
