import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import math






class Critic(nn.Module):
    def __init__(self,input_dim,dis_dims, pac=1):
        super(Critic, self).__init__()
        dim = input_dim*pac
        self.pac = pac
        self.pacdim = dim
        seq =[]
        for item in list(dis_dims):
            seq +=[
                nn.Linear(dim,item),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5)
            ]
            dim = item
        seq +=[nn.Linear(dim,1)]
        self.seq = nn.Sequential(*seq)    
       
        
        
    def forward(self,input):
        assert input.size()[0]% self.pac == 0
        return self.seq(input.view(-1, self.pacdim))

class Residual(nn.Module):
    def __init__(self, i, o):
        super(Residual,self).__init__()
        self.fc = nn.Linear(i,o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self,input):
        out = self.relu(self.bn(self.fc(input)))
        return torch.cat([out,input],dim =1)


class Generator(nn.Module):
    def __init__(self, embedding_dim,gen_dim, data_dim):
        super(Generator, self).__init__()

        dim = embedding_dim
        seq = []
        for item in list(gen_dim):
            seq +=[
                Residual(dim,item)
            ]
            dim +=item
        seq.append(nn.Linear(dim,data_dim))
        self.seq = nn.Sequential(*seq)   
        
    def forward(self,input):
        data = self.seq(input)
        return data



def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1]=='tanh':
            ed = st + item[0]
            data_t.append(torch.sigmoid(data[:, st:ed]))  #此处修改了tanh成sigmoid
            st = ed 
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:,st:ed], tau=0.2))
            st = ed
        else:
            assert 0
    return torch.cat(data_t, dim=1)


    
def gradient_penalty_compute(netD,real_data,fake_data,device='cpu',pac=1,lambda_ = 10):

    alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
    alpha = alpha.repeat(1, pac, real_data.size(1))
    alpha = alpha.view(-1, real_data.size(1))

    interpolates = (alpha*real_data + (1-alpha)*fake_data).requires_grad_(True)

    d_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size(),device=device),
        create_graph=True,
        retain_graph=True, 
        only_inputs=True)[0]

    gradient_penalty = (
        (gradients.view(-1, pac*real_data.size(1)).norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty
