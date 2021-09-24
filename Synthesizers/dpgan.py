import numpy as np
import torch
import torch.optim as optim
import time
from Synthesizers.model import Critic,Generator,apply_activate
from torch.utils.data import DataLoader, TensorDataset
from Synthesizers.utils import set_seed,get_column_loc
from Synthesizers.rdp_accountant import compute_rdp, get_privacy_spent





class dp_WGanSynthesizer():
    def __init__(self, 
                 embedding_dim = 128,
                 gen_dim =(256, 256),
                 dis_dim = (256,256),
                 l2scale = 1e-6,
                 batch_size = 100,
                 epochs = 150,
                 sigma = 1.0,
                 seed = 0
                 ):
        self.embedding_dim = embedding_dim
        self.dis_dim = dis_dim
        self.gen_dim = gen_dim
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sigma = sigma
        self.seed = seed
        if sigma > 0:
            self.cg = 0.1
            self.noise_multiplier = self.sigma/self.cg
            print("noise_multiplier:", self.noise_multiplier) 
        else: 
            print("No DP")
    
    def fit(self,data,data_info,target_eps=1.0,target_delta=1e-5):

        set_seed(self.seed)
        self.data_info = data_info
        data_dim = data.shape[1]
        print("Privacy budget:(%.5f, %.5f)" % (target_eps,target_delta))
        print("Privacy tuple:(%.5f, %.5f)" % (self.cg,self.sigma))
        print("data dim:", data_dim)
        self.ganinput = torch.from_numpy(data.astype('float32')).to(self.device)
        dataset = TensorDataset(self.ganinput)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,num_workers=0)

        #initialize models and optimizers
        self.myG = Generator(self.embedding_dim, self.gen_dim, data_dim).to(self.device)
        myD = Critic(data_dim, self.dis_dim).to(self.device)
        optimG = optim.Adam(self.myG.parameters(),lr = 1e-4, betas = (0.5,0.9), weight_decay=self.l2scale)
        optimD = optim.Adam(myD.parameters(), lr=1e-4, betas = (0.5,0.9) )
        
        
        mean = torch.zeros(self.batch_size,self.embedding_dim,device = self.device)
        std = mean +1


        self.tanh_list,self.soft_list = get_column_loc(self.data_info)

        print("Begin training...")
        print("Number of iterations each epochs:", len(loader))

        steps = 0 
        for i in range(self.epochs):
            torch.cuda.synchronize()
            start = time.time()
            for _, data in enumerate(loader):

                for _ in range(1):
                    
                    #train D with real samples
                    real = data[0]
                    y_real = myD(real)

                    #train D with fake samples
                    noise = torch.normal(mean = mean, std = std) 
                    fake = self.myG(noise)  
                    fakeact = apply_activate(fake, self.tanh_list,self.soft_list)
                    y_fake = myD(fakeact.detach())

                    loss_d = -torch.mean(y_real) + torch.mean(y_fake)
                    optimD.zero_grad()
                    loss_d.backward() 

                    
                    #add DP noise
                    if self.sigma>0:
                        #clip gradient
                        torch.nn.utils.clip_grad_norm_(myD.parameters(), self.cg)
                        #add noise
                        for name, params in myD.named_parameters():
                            gaussian = torch.randn(params.grad.shape).to(self.device)* (self.sigma**2) / self.batch_size
                            params.grad = params.grad + gaussian.detach()


                        # Calculate the current privacy cost using the accountant
                        steps += 1
                    
                        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
                        rdp = compute_rdp(q=self.batch_size / self.ganinput.shape[0],
                                        noise_multiplier=self.noise_multiplier, 
                                        steps=steps, 
                                        orders=orders)
                        epsilon, _, _ = get_privacy_spent(orders, rdp, target_delta=target_delta)

                    optimD.step()
                    for p in myD.parameters(): p.data.clamp_(-0.01, 0.01)
                    
                #train the generator 
                noise_2 = torch.normal(mean = mean, std = std)
                fake_2 = self.myG(noise_2)
                fakeact_2 = apply_activate(fake_2, self.tanh_list,self.soft_list)
                
                y_fake_2 = myD(fakeact_2)
                loss_g = -torch.mean(y_fake_2)

                optimG.zero_grad()
                loss_g.backward()
                optimG.step()
   
            torch.cuda.synchronize()   
            end = time.time()
            diff = end-start
            
            self.dp_epoch = i
            if self.sigma>0:
                print('[%d/%d]\t  Loss_D: %.4f\tLoss_G: %.4f\t runtime: %.4f s\t Current eps: %.4f \t' % (i, self.epochs,
                    loss_d.item(), loss_g.item(),diff, epsilon))
            else:
                print('[%d/%d]\t  Loss_D: %.4f\tLoss_G: %.4f\t runtime: %.4f s\t' % (i, self.epochs,
                    loss_d.item(), loss_g.item(),diff))

    
    
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




