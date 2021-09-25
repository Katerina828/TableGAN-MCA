import numpy as np
import torch
import pandas as pd
from data_preprocess.transformer import BaseTransformer
from Synthesizers.utils import compas_postprocess,lawsch_postprocess,adult_postprocess,set_seed
from Synthesizers.wgangp import apply_activate

def train_Gmodel(model,data,data_info, seed,epochs,bs):
    set_seed(seed)
    tf = BaseTransformer(data,data_info)
    training = tf.transform()
    gan = model(epochs = epochs,seed=seed,batch_size = bs)
    gan.fit(training,data_info)
    return gan,tf

def sample_data(n,generator,seed,data_info, bs=500):
    print("Begin sample，seed=",seed)
    set_seed(seed)
    steps = n // bs +1
    data = []
    for _ in range(steps):
        noise = torch.randn(bs , 128)
        fake = generator(noise)
        f_samples = apply_activate(fake, data_info)
        fakeact = f_samples 
                
        data.append(fakeact.detach().numpy())  
    data = np.concatenate(data, axis=0) 
    sampled_data = data[:n]  
    return sampled_data

def sample_data_fromGAN(gan,tf,n,seed):
    sample = gan.sample(n = n,seed=seed)
    data_inverse = tf.inverse_transform(sample)
    syn = pd.DataFrame(data_inverse,index=None,columns = tf.columns_name)

    syn = adult_postprocess(syn)

    return syn


def sample_data_fromGen(Gen,tf,n,data_info,seed):
    
    sample = sample_data(n,Gen,seed,data_info, bs=500)
    data_inverse = tf.inverse_transform(sample)
    syn = pd.DataFrame(data_inverse,index=None,columns = tf.columns_name)
    syn = adult_postprocess(syn)

    return syn
