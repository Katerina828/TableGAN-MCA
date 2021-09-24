import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

from Synthesizers.model import Critic,Generator,apply_activate,gradient_penalty_compute
from torch.utils.data import DataLoader, TensorDataset
import time
from Synthesizers.utils import set_seed,get_column_loc





class constrained_WGANGP():
    def __init__(self, 
                 embedding_dim = 128,
                 gen_dim =(256, 256),
                 dis_dim = (256,256),
                 l2scale = 1e-6,
                 batch_size = 100,
                 epochs = 150,
                 seed = 0
                 ):
        self.embedding_dim = embedding_dim
        self.dis_dim = dis_dim
        self.gen_dim = gen_dim
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = seed
    
    def fit(self,data,data_info,defence_tag=True):
        set_seed(self.seed)
        #data need to be transformed
        self.data_info = data_info
        data_dim = data.shape[1]
        print("data dim:", data_dim)
        self.ganinput = torch.from_numpy(data.astype('float32')).to(self.device)


        dataset = TensorDataset(self.ganinput)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,num_workers=0)

        #initialize models and optimizers
        self.myG = Generator(self.embedding_dim, self.gen_dim, data_dim).to(self.device)
        self.myD = Critic(data_dim, self.dis_dim).to(self.device)
        optimG = optim.Adam(self.myG.parameters(),lr = 1e-4, betas = (0.5,0.9), weight_decay=self.l2scale)
        optimD = optim.Adam(self.myD.parameters(), lr=1e-4, betas = (0.5,0.9) )
        
        if defence_tag == True:
            self.bs_redundancy = 900
        else:
            self.bs_redundancy = self.batch_size 
        
        mean = torch.zeros(self.bs_redundancy,self.embedding_dim,device = self.device)
        std = mean +1


        self.tanh_list,self.soft_list = get_column_loc(self.data_info)

        
        ##### ----defense part---- ####
        if defence_tag == True:
            bucket_boundaries = np.array(
                [0.        , 0.01666667, 0.03333333, 0.05      , 0.06666667,
                0.08333333, 0.1       , 0.11666667, 0.13333333, 0.15      ,
                0.16666667, 0.18333333, 0.2       , 0.21666667, 0.23333333,
                0.25      , 0.26666667, 0.28333333, 0.3       , 0.31666667,
                0.33333333, 0.35      , 0.36666667, 0.38333333, 0.4       ,
                0.41666667, 0.43333333, 0.45      , 0.46666667, 0.48333333,
                0.5       , 0.51666667, 0.53333333, 0.55      , 0.56666667,
                0.58333333, 0.6       , 0.61666667, 0.63333333, 0.65      ,
                0.66666667, 0.68333333, 0.7       , 0.71666667, 0.73333333,
                0.75      , 0.76666667, 0.78333333, 0.8       , 0.81666667,
                0.83333333, 0.85      , 0.86666667, 0.88333333, 0.9       ,
                0.91666667, 0.93333333, 0.95      , 0.96666667, 0.98333333,
                1.        ])

            c,f,c2,f2 = construct_coefficients(samplesize=self.bs_redundancy,
                                                dim =data_dim ,
                                                trainingsize =data.shape[0], 
                                                device=self.device)
            print("Redundant Batch Size:", self.bs_redundancy)
            training = self.ganinput*c2 +f2
            #bucketization
            #training[:,1:2] = bucketize(training[:,1:2], bucket_boundaries)
            training = torch.round(training)
        ##### -----defense part----- ####
        None_overlapping = []
        print("Begin training...")
        print("Number of iterations each epochs:", len(loader))
        for i in range(self.epochs):
            torch.cuda.synchronize()
            start = time.time()
            for _, data in enumerate(loader):

                for _ in range(2):
                    
                    real = data[0]
                    noise = torch.normal(mean = mean, std = std) 
                    fake = self.myG(noise)  
                    

                    f_samples = apply_activate(fake, self.tanh_list,self.soft_list)  # apply activation
                    
                    if defence_tag == True: 
                    #choose fake samples that disjoint with the training set
                        choose_index = choose_disjoint(f_samples,training, self.soft_list,self.tanh_list,self.device,c,f)
                        assert len(choose_index)>=self.batch_size, "Insufficient samples"
                        fakeact = f_samples[choose_index][:self.batch_size]
                    else:
                        fakeact = f_samples
                    y_real = self.myD(real)
                    y_fake = self.myD(fakeact.detach())
                    pen = gradient_penalty_compute(self.myD, real, fakeact, self.device)
                    loss_d = -torch.mean(y_real) + torch.mean(y_fake) + pen
                    optimD.zero_grad()
                    loss_d.backward() 
                    optimD.step()
                    torch.cuda.synchronize()       
                    
                noise_2 = torch.normal(mean = mean, std = std)
                fake_2 = self.myG(noise_2)
                f_samples2 = apply_activate(fake_2, self.tanh_list,self.soft_list)
                if defence_tag == True:
                    choose_index2 = choose_disjoint(f_samples2,training, self.soft_list,self.tanh_list,self.device,c,f)
                    assert len(choose_index2)>=self.batch_size, "Insufficient samples"
                    fakeact_2 = f_samples2[choose_index2][:self.batch_size]
                else:
                    fakeact_2 = f_samples2
                y_fake_2 = self.myD(fakeact_2)
                loss_g = -torch.mean(y_fake_2)

                optimG.zero_grad()
                loss_g.backward()
                optimG.step()
            
            torch.cuda.synchronize()   
            end = time.time()
            diff = end-start
            if defence_tag == True:
                print('[%d/%d]\t  Loss_D: %.4f\tLoss_G: %.4f\t runtime: %.4f s\t Non-overlapping size: %d' % (i, self.epochs,
                    loss_d.item(), loss_g.item(),diff,len(choose_index)))
                None_overlapping.append(len(choose_index))

    
    def sample(self,n,seed=0):
        print("Begin sample，seed=",seed)
        set_seed(seed)
        steps = n // self.batch_size +1
        data = []
        for _ in range(steps):

            noise = torch.randn(self.batch_size , self.embedding_dim).to(self.device)
            fake = self.myG(noise)
            fakeact = apply_activate(fake,  self.tanh_list,self.soft_list)
            data.append(fakeact.detach().cpu().numpy())  
        data = np.concatenate(data, axis=0)  
        self.data = data[:n] 
        return self.data     
    




def bucketize(tensor, bucket_boundaries):
    result = torch.zeros_like(tensor, dtype=torch.int32)
    for boundary in bucket_boundaries:
        result += (tensor > boundary).int()
    return result

def choose_disjoint_lawsch(fake_sample,training,soft_list,tanh_list,device,c,f):

    bucket_boundaries = np.array(
    [0.        , 0.01666667, 0.03333333, 0.05      , 0.06666667,
       0.08333333, 0.1       , 0.11666667, 0.13333333, 0.15      ,
       0.16666667, 0.18333333, 0.2       , 0.21666667, 0.23333333,
       0.25      , 0.26666667, 0.28333333, 0.3       , 0.31666667,
       0.33333333, 0.35      , 0.36666667, 0.38333333, 0.4       ,
       0.41666667, 0.43333333, 0.45      , 0.46666667, 0.48333333,
       0.5       , 0.51666667, 0.53333333, 0.55      , 0.56666667,
       0.58333333, 0.6       , 0.61666667, 0.63333333, 0.65      ,
       0.66666667, 0.68333333, 0.7       , 0.71666667, 0.73333333,
       0.75      , 0.76666667, 0.78333333, 0.8       , 0.81666667,
       0.83333333, 0.85      , 0.86666667, 0.88333333, 0.9       ,
       0.91666667, 0.93333333, 0.95      , 0.96666667, 0.98333333,
       1.        ])

    #continous features
    temp = fake_sample[:,tanh_list[0]:(tanh_list[-1]+1)]

    #catagorical features
    for i in range(len(soft_list)-1):
        tem_data = fake_sample[:,soft_list[i]:soft_list[i+1]]
        x = torch.zeros_like(tem_data,device = device)
        index = torch.argmax(tem_data,dim=1)
        x[np.arange(0,fake_sample.shape[0],1),index]=1
        temp = torch.cat([temp,x],1)
    tem_data = fake_sample[:,soft_list[-1]:]
    x = torch.zeros_like(tem_data,device = device)
    tem_soft = torch.argmax(tem_data,dim=1)
    x[np.arange(0,fake_sample.shape[0],1),tem_soft]=1
    temp = torch.cat([temp,x],1)
    
    #transform age   
    temp =  temp* c + f
    
    #bucketization gpa
    temp[:,1:2] = bucketize(temp[:,1:2], bucket_boundaries)
    temp = torch.round(temp)
    #training = torch.round(training*c2 +f2)
    
    concat = torch.cat((temp,training), axis=0)
    _,inverse_index,count = torch.unique(concat,dim=0, return_inverse=True,return_counts=True)
    x = ~(count-1)[inverse_index][:temp.shape[0]].bool()
    choose_index = torch.nonzero(x).squeeze()
    
    return choose_index

def choose_disjoint(fake_sample,training,soft_list,tanh_list,device,c,f):

    #continous features
    temp = fake_sample[:,tanh_list[0]:(tanh_list[-1]+1)]

    #catagorical features
    for i in range(len(soft_list)-1):
        tem_data = fake_sample[:,soft_list[i]:soft_list[i+1]]
        x = torch.zeros_like(tem_data,device = device)
        index = torch.argmax(tem_data,dim=1)
        x[np.arange(0,fake_sample.shape[0],1),index]=1
        temp = torch.cat([temp,x],1)
    tem_data = fake_sample[:,soft_list[-1]:]
    x = torch.zeros_like(tem_data,device = device)
    tem_soft = torch.argmax(tem_data,dim=1)
    x[np.arange(0,fake_sample.shape[0],1),tem_soft]=1
    temp = torch.cat([temp,x],1)
    
    #transform age   
    temp = torch.round( temp* c + f)
    
    #training = torch.round(training*c2 +f2)
    
    concat = torch.cat((temp,training), axis=0)
    _,inverse_index,count = torch.unique(concat,dim=0, return_inverse=True,return_counts=True)
    x = ~(count-1)[inverse_index][:temp.shape[0]].bool()
    choose_index = torch.nonzero(x).squeeze()
    
    return choose_index

def construct_coefficients(samplesize,dim,trainingsize,device):
    a = torch.tensor([73.])
    a = a.repeat(samplesize,1)
    b= torch.ones([samplesize,dim-1])
    c = torch.cat([a,b],1).to(device)
    
    d = torch.tensor([19.])
    d = d.repeat(samplesize,1)
    e= torch.zeros_like(b)
    f = torch.cat([d,e],1).to(device)

    a2 = torch.tensor([73.])
    a2 = a2.repeat(trainingsize,1)
    b2= torch.ones([trainingsize,dim-1])
    c2 = torch.cat([a2,b2],1).to(device)
    
    d2 = torch.tensor([19.])
    d2 = d2.repeat(trainingsize,1)
    e2 = torch.zeros_like(b2)
    f2 = torch.cat([d2,e2],1).to(device)
    return c,f,c2,f2

def construct_coefficients_lawsch(samplesize,dim,trainingsize,device):
    a = torch.tensor([64.])
    a = a.repeat(samplesize,1)
    b= torch.ones([samplesize,dim-1])
    c = torch.cat([a,b],1).to(device)
    
    d = torch.tensor([120.])
    d = d.repeat(samplesize,1)
    e= torch.zeros_like(b)
    f = torch.cat([d,e],1).to(device)

    a2 = torch.tensor([64.])
    a2 = a2.repeat(trainingsize,1)
    b2= torch.ones([trainingsize,dim-1])
    c2 = torch.cat([a2,b2],1).to(device)
    
    d2 = torch.tensor([120.])
    d2 = d2.repeat(trainingsize,1)
    e2 = torch.zeros_like(b2)
    f2 = torch.cat([d2,e2],1).to(device)
    return c,f,c2,f2


def sample_adult(n,generator,seed,ganinput,defense_tag,tanh_list,soft_list,samplesize = 900, bs=500):
    if defense_tag == True:
        c,f,c2,f2 = construct_coefficients(samplesize=samplesize,
                                            dim =ganinput.shape[1] ,
                                            trainingsize =ganinput.shape[0], 
                                            device='cpu')
    
        ganinput = torch.from_numpy(ganinput.astype('float32'))
        training = torch.round(ganinput*c2 +f2)
    print("Begin sample，seed=",seed)
    set_seed(seed)
    steps = n // bs +1
    data = []
    for _ in range(steps):
        noise = torch.randn(samplesize , 128)
        fake = generator(noise)
        f_samples = apply_activate(fake, tanh_list,soft_list)
        
        if defense_tag == True:
            choose_index = choose_disjoint(f_samples,training, soft_list,tanh_list,'cpu',c,f)
            assert len(choose_index)>=bs, "Insufficient samples"
            fakeact = f_samples[choose_index][:bs]
        else:
            fakeact = f_samples 
                
        data.append(fakeact.detach().numpy())  
    data = np.concatenate(data, axis=0)  
    sampled_data = data[:n]  
    return sampled_data

def sample_lawsch(n,generator,seed,ganinput,defense_tag,tanh_list,soft_list,samplesize = 900, bs=500):
    if defense_tag==True:
        bucket_boundaries = np.array(
                [0.        , 0.01666667, 0.03333333, 0.05      , 0.06666667,
                0.08333333, 0.1       , 0.11666667, 0.13333333, 0.15      ,
                0.16666667, 0.18333333, 0.2       , 0.21666667, 0.23333333,
                0.25      , 0.26666667, 0.28333333, 0.3       , 0.31666667,
                0.33333333, 0.35      , 0.36666667, 0.38333333, 0.4       ,
                0.41666667, 0.43333333, 0.45      , 0.46666667, 0.48333333,
                0.5       , 0.51666667, 0.53333333, 0.55      , 0.56666667,
                0.58333333, 0.6       , 0.61666667, 0.63333333, 0.65      ,
                0.66666667, 0.68333333, 0.7       , 0.71666667, 0.73333333,
                0.75      , 0.76666667, 0.78333333, 0.8       , 0.81666667,
                0.83333333, 0.85      , 0.86666667, 0.88333333, 0.9       ,
                0.91666667, 0.93333333, 0.95      , 0.96666667, 0.98333333,
                1.        ])
    
        c,f,c2,f2 = construct_coefficients_lawsch(samplesize=samplesize,
                                            dim =ganinput.shape[1] ,
                                            trainingsize =ganinput.shape[0], 
                                            device='cpu')
    
        ganinput = torch.from_numpy(ganinput.astype('float32'))
        training = ganinput*c2 +f2
        training[:,1:2] = bucketize(training[:,1:2], bucket_boundaries)
        training = torch.round(training)
        
    print("Begin sample，seed=",seed)
    set_seed(seed)
    steps = n // bs +1
    data = []
    for _ in range(steps):
        noise = torch.randn(samplesize , 128)
        fake = generator(noise)
        f_samples = apply_activate(fake, tanh_list,soft_list)
        
        if defense_tag == True:
            choose_index = choose_disjoint_lawsch(f_samples,training, soft_list,tanh_list,'cpu',c,f)
            assert len(choose_index)>=bs, "Insufficient samples"
            fakeact = f_samples[choose_index][:bs]
        else:
            fakeact = f_samples 
                
        data.append(fakeact.detach().numpy())  
    data = np.concatenate(data, axis=0) 
    sampled_data = data[:n] 
    return sampled_data