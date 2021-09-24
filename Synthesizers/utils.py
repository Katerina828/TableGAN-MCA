import numpy as np
import torch
import pandas as pd
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic =True
    random.seed(seed)




#data：pandas dataframe
def get_data_info(data,categorical_columns):
    data_info = []
    con_columns = [col for col in data.columns if col not in categorical_columns]
    for column in con_columns:
        data_info.append((1,'tanh', column))
    for column in categorical_columns:
        a = (data[column].unique().shape[0],'softmax',column)
        data_info.append(a)

    return data_info  


def get_column_loc(data_info):
    tanh_list = []
    soft_list =[]
    st=0
    for item in data_info:
        if item[1]=='tanh':
            tanh_list.append(st)
            st = st + item[0]
        elif item[1]=='softmax':
            soft_list.append(st)
            st = st + item[0]
        else:
            assert 0
    return tanh_list,soft_list

          

def lawsch_postprocess(data):
    data['lsat'] = data['lsat'].astype('float').round(decimals=0)
    data['gpa'] = data['gpa'].astype('float')
    data['gpa'] = data['gpa'].round(decimals=2)
    return data

def compas_postprocess(data):
    data.loc[(data["diff_jail"] < 0 ,"diff_jail")]= 0
    data[['age','diff_custody','diff_jail','priors_count']] = data[['age','diff_custody','diff_jail','priors_count']].astype('float')
    data[['age','diff_custody','diff_jail','priors_count']] = data[['age','diff_custody','diff_jail','priors_count']].round(decimals=0)

    return data

def adult_postprocess(data):
    data['Age'] = data['Age'].astype('float').round(decimals=0)
    data['EducationNum'] = data['EducationNum'].astype('float64')
    data['HoursPerWeek'] = data['HoursPerWeek'].astype('float64')
    return data
