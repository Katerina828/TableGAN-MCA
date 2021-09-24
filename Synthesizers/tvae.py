import numpy as np
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from Synthesizers.base import BaseSynthesizer
from Synthesizers.utils import set_seed, get_data_info
from data_preprocess.transformer import BaseTransformer

class Encoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item
        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input):
        feature = self.seq(input)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input):
        return self.seq(input), self.sigma


def loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            std = sigmas[st]
            loss.append(((x[:, st] - torch.tanh(recon_x[:, st])) ** 2 / 2 / (std ** 2)).sum())
            loss.append(torch.log(std) * x.size()[0])
            st = ed

        elif item[1] == 'softmax':
            ed = st + item[0]
            loss.append(item[0]/2*cross_entropy(
                recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
            #loss.append(F.binary_cross_entropy(recon_x[:, st:ed],x[:, st:ed], reduction='sum'))
            st = ed

        else:
            assert 0
    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class TVAESynthesizer(BaseSynthesizer):
    """TVAESynthesizer."""

    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        seed = 0
    ):

        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.seed = seed
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = 2
        self.epochs = epochs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, data,data_info):
        set_seed(self.seed)
        self.data_info = data_info
        self.vaeinput = torch.from_numpy(data.astype('float32')).to(self.device)
        dataset = TensorDataset(self.vaeinput)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        data_dim = data.shape[1]
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self.device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, data_dim).to(self.device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)

        for i in range(self.epochs):
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self.device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = loss_function(
                    rec, real, sigmas, mu, logvar, self.data_info , self.loss_factor)
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

            print('[Epoch:{}/{}]\t Loss_D: {:.4f}'.format (i, self.epochs, loss))


    def sample(self, n,seed=0):
        print("Begin sample...")
        set_seed(seed)
        steps = n // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self.device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return data


def train_vae(data,cat_vars,seed,epochs,bs):
    data_info = get_data_info(data,cat_vars)
    tf = BaseTransformer(data,data_info)
    vaeinput= tf.transform()
    vae = TVAESynthesizer(epochs = epochs,seed=seed,batch_size = bs)
    vae.fit(vaeinput,data_info)
    return vae, tf 