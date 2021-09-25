import numpy as np
from sklearn.preprocessing import OrdinalEncoder,KBinsDiscretizer
import pandas as pd
from sklearn.neural_network import MLPClassifier
from collections import Counter

from Synthesizers.utils import get_data_info
from Synthesizers.train import train_Gmodel,sample_data_fromGAN

def train_classifier(train):
    train = train.drop(columns=['counts_in_real'])
    y_train = train.pop('label')
    X_train = train.values
    classifier = MLPClassifier(hidden_layer_sizes=(100, ),max_iter = 200)
    classifier.fit(X_train, y_train)
    
    return classifier

def predict_prob(model,test):
    test  = test.drop(columns=['counts_in_real'])
    test.pop('label')
    X_test = test.values
    y_score = model.predict_proba(X_test)[:,1]
    return y_score

def unique_exam(data):
    uniques,counts = np.unique(data.values,axis=0,return_counts=True)
    return Counter(counts).most_common(80), uniques

def compute_intersection(data1,data2):
        x = np.concatenate((data1, data2), axis=0)
        _,index,counts = np.unique(x,axis=0,return_index = True,return_counts=True)
        idx = []
        for i,count in enumerate(counts):
            if count==2:
                idx.append(index[i])
        duplicate = x[idx]
        return duplicate
        
def add_true_label(real,fake):
    unique, _,_ = np.unique(real,axis=0, return_inverse=True, return_counts=True)
    print("real unique number:", unique.shape)
    unique, inverse_index,count = np.unique(fake,axis=0, return_inverse=True, return_counts=True)
    print("fake unique number:",unique.shape)
    fake_new = pd.DataFrame(unique[inverse_index],columns = real.columns)
    fake_new['freq'] = count[inverse_index].reshape(-1,1)

    real_con_fake  = np.concatenate((fake,real),axis=0)
    _, inverse_index2,count2 = np.unique(real_con_fake,axis=0, return_inverse=True, return_counts=True)
    fake_new['counts_in_real'] = count2[inverse_index2].reshape(-1,1)[:fake_new.shape[0]] - count[inverse_index].reshape(-1,1)

    _,uniques_real = unique_exam(real)
    #add real label
    real_part = compute_intersection(unique,uniques_real)
    test_data = np.concatenate((unique,real_part), axis=0) 
    _,count2 = np.unique(test_data,axis=0, return_counts=True)
    d_count = count2-1
    real_label = (d_count[inverse_index] >0)
    fake_new['label'] = real_label
    fake_new['label'] = fake_new['label'].astype('int')
    return fake_new

def label_func(real, syn):

    enc = OrdinalEncoder()
    colomns = ["WorkClass","MaritalStatus","Occupation","Relationship","Race","Gender","CapitalGain","CapitalLoss","NativeCountry","Income"]
    real[colomns] = enc.fit_transform(real[colomns])
    syn[colomns] = enc.transform(syn[colomns])
    synlabeled = add_true_label(real,syn)
    print("Number of positive synthetic data:",synlabeled.label.value_counts()[1])
    print("Positive percentage:",synlabeled.label.value_counts()[1]/len(synlabeled))
    return synlabeled





def Tablegan_mca(real,syn,data_info,shadowmodel,epochs,seed=0):
    epochs=300
    bs=500

    #train shadow model and sample shadow datasets
    synstate =syn.copy()
    
    cat_vars = [data_info[i][2] for i in range(len(data_info)) if data_info[i][1]=='softmax']
    data_info = get_data_info(syn,cat_vars)
    print("Synthetic datasets info:", data_info)


    smodel,tf = train_Gmodel(shadowmodel,syn,data_info,seed=seed,epochs=epochs,bs=bs)
    shadow = sample_data_fromGAN(smodel,tf,n=len(syn),seed=seed)
    labeled_shadow = label_func(synstate, shadow)  #training data
    
    
    labeled_syn = label_func(real, syn) #testing data
    classifier= train_classifier(labeled_shadow) #train attack model 
    y_score = predict_prob(classifier,labeled_syn)
    labeled_syn['y_score']=y_score

    return labeled_syn


    
    
    