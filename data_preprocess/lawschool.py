import numpy as np
import pandas as pd
from os import path
from Synthesizers.utils import get_data_info
from sklearn.model_selection import train_test_split

def lawsch_preprocess(dataset):
    dataset.drop(['enroll', 'asian', 'black', 'hispanic', 'white', 'missingrace', 'urm'], axis=1, inplace=True)
    dataset.dropna(axis=0, inplace=True, subset=['admit'])
    dataset.replace(to_replace='', value=np.nan, inplace=True)
    dataset.dropna(axis=0, inplace=True)
    dataset = dataset[dataset['race'] != 'Asian']

    for col in dataset.columns:
        if dataset[col].isnull().sum() > 0:
            dataset.drop(col, axis=1, inplace=True)

    con_vars = ['lsat','gpa']
    cat_vars = [col for col in dataset.columns if col not in con_vars]
    return dataset,cat_vars


def load_lawsch():
    data_dir = 'dataset'
    data_file = path.join(data_dir, 'lawschs1_1.dta')
    df = pd.read_stata(data_file)

    data, cat_vars = lawsch_preprocess(df)
    data_info = get_data_info(data ,cat_vars)
    print("Data info:", data_info) 

    train_data, remained_data = train_test_split(data, test_size=0.5, random_state=2)
    _, test_data = train_test_split(remained_data, test_size=3/7, random_state=2)
    
    return train_data, test_data, data_info,cat_vars