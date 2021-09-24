import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from Synthesizers.utils import set_seed



class VAE(nn.Module):
    def __init__(self,embedding_dim,data_dim):
        super().__init__()
        self.data_dim=data_dim
        self.fc1 = nn.Linear(data_dim, 128)
        self.fc2 = nn.Linear(128,128)
        self.mu = nn.Linear(128, embedding_dim)
        self.var = nn.Linear(128, embedding_dim)

        self.fc3 = nn.Linear(embedding_dim,128)
        self.fc4 = nn.Linear(128,128)
        self.out = nn.Linear(128,data_dim)

    def encoder(self,x):
        feature =F.relu(self.fc2(F.relu(self.fc1(x)))) 
        mu = self.mu(feature)
        var = self.var(feature)
        return mu, var
    
    def reparameterize(self,mu,log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decoder(self,z):
        hidden = F.relu(self.fc4(F.relu(self.fc3(z))))
        output = self.out(hidden)
        return torch.sigmoid(output)

    def forward(self,x):

        z_mu,log_var = self.encoder(x)
        z = self.reparameterize(z_mu,log_var)
        recon_x = self.decoder(z)
        return recon_x, z_mu, log_var





class vanilla_VAE():
    """TVAESynthesizer."""

    def __init__(
        self,
        embedding_dim=128,
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        seed = 0
    ):

        self.embedding_dim = embedding_dim
        self.seed = seed
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self,data,data_info):
        print("train vanillaVAE...")
        
        set_seed(self.seed)
        self.data_info = data_info
        data_dim = data.shape[1]
        print("data dim:", data_dim)
        

        self.vaeinput = torch.from_numpy(data.astype('float32')).to(self.device)
        dataset = TensorDataset(self.vaeinput)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.vae = VAE(self.embedding_dim,data_dim).to(self.device)

        optimizer = optim.Adam(self.vae.parameters(),weight_decay=self.l2scale)

        print('begin training...')
        loss_vector =[]
        kl_loss_vector =[]
        recon_loss_vector = []

        for i in range(self.epochs):   
            self.vae.train()
            train_loss = 0
            train_kl_loss = 0
            train_recon_loss = 0
            for _, data in enumerate(loader):

                real = data[0].to(self.device)
                optimizer.zero_grad()
                recon_x,z_mu, log_var  = self.vae.forward(real)
                self.mu = z_mu
                self.log_var = log_var
                #recon_x = apply_activate(out,self.data_info)

                kl_loss = -0.5 * torch.sum(1+log_var - z_mu.pow(2)-log_var.exp())
                #real_b = torch.bernoulli(real)
                #recon_loss = F.binary_cross_entropy(recon_x,real,reduction='sum')
                recon_loss = loss_function(recon_x,real,self.data_info)

                loss =  kl_loss + 2*recon_loss
                loss.backward()
                train_kl_loss += kl_loss.item()
                train_recon_loss += recon_loss.item()
                train_loss += loss.item()
                optimizer.step()

            print('[Epoch:{}/{}]\t Loss_D: {:.4f}'.format (i, self.epochs, train_loss/ len(loader.dataset)))
            
            kl_loss_vector.append(train_kl_loss/len(loader.dataset))
            recon_loss_vector.append(train_recon_loss/len(loader.dataset))
            loss_vector.append( train_loss/ len(loader.dataset))


        return kl_loss_vector,recon_loss_vector,loss_vector

    def sample(self, n,seed=0):
        print("Begin sample...")
        set_seed(seed)
        steps = n // self.batch_size + 1
        data = []
        with torch.no_grad():
            for _ in range(steps):
                
                noise = torch.randn(self.batch_size , self.embedding_dim ).to(self.device)
                fake = self.vae.decoder(noise)
                data.append(fake.cpu().numpy())

        data = np.concatenate(data, axis=0)
        self.data = data[:n]

        return self.data 
        #return self.data
        


def loss_function(recon_x,x,data_info):
    st= 0
    loss =[]
    for info in data_info:
        if info[1]=='tanh':
            ed = st + info[0]
            loss.append( F.mse_loss(recon_x[:,st:ed], x[:,st:ed],reduction='sum'))
            st = ed
        elif info[1] == 'softmax':
            ed = st + info[0]
            loss.append(F.cross_entropy(recon_x[:,st:ed],torch.argmax(x[:,st:ed], dim=-1),reduction='sum'))
            st = ed

        else:
            assert 0
    return sum(loss)


