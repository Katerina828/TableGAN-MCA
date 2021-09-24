import pandas as pd
from os import path
from sklearn.model_selection import train_test_split
from Synthesizers.utils import get_data_info
from sklearn.model_selection import train_test_split

#-----------------------------
#Compas: orginal :(7214, 53)
# After preproces: (5278, 11)
# predict two_year_recid, 
# Task: binary classification

def compas_preprocess(df):

    df = df[df['days_b_screening_arrest'] >= -30]
    df = df[df['days_b_screening_arrest'] <= 30]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != '0']
    df = df[df['score_text'] != 'N/A']

    df['in_custody'] = pd.to_datetime(df['in_custody'])
    df['out_custody'] = pd.to_datetime(df['out_custody'])
    df['diff_custody'] = (df['out_custody'] - df['in_custody']).dt.days
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['diff_jail'] = (df['c_jail_out'] - df['c_jail_in']).dt.days

    df.drop(
        [
            'id', 'name', 'first', 'last', 'v_screening_date', 'compas_screening_date', 'dob', 'c_case_number',
            'screening_date', 'in_custody', 'out_custody', 'c_jail_in', 'c_jail_out'
        ], axis=1, inplace=True
    )
    df = df[df['race'].isin(['African-American', 'Caucasian'])]

    features = df.drop(['is_recid', 'is_violent_recid', 'violent_recid', 'two_year_recid'], axis=1)
    labels = 1 - df['two_year_recid']

    features = features[[
        'age', 'sex', 'race', 'diff_custody', 'diff_jail', 'priors_count', 'juv_fel_count', 'c_charge_degree',
        'v_score_text'
    ]]

    data = pd.concat([features,labels],axis = 1)
    data[['juv_fel_count','two_year_recid']] = data[['juv_fel_count','two_year_recid']].astype('object')
    con_vars = [i for i in data.columns if data[i].dtype=='int64'or data[i].dtype=='float64']
    cat_vars = [col for col in data.columns if col not in con_vars]
    return data, cat_vars


def load_compas():
    data_dir = 'dataset'
    data_file = path.join(data_dir,'compas-scores-two-years.csv')
    df = pd.read_csv(data_file)
    data, cat_vars = compas_preprocess(df)
    data_info = get_data_info(data ,cat_vars)
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=2)
    print("Data info:", data_info) 
    return train_data, test_data,data_info,cat_vars

