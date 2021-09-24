import pandas as pd
from Synthesizers.utils import get_data_info
import operator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer


#CapitalGain 121 --->3  CapitalLoss  97--->3
def CapitalGainLoss(data):

    data.loc[(data["CapitalGain"] > 7647.23),"CapitalGain"] = 'high'
    data.loc[(data["CapitalGain"] == 0 ,"CapitalGain")]= 'zero'
    data.loc[operator.and_(data["CapitalGain"]!='zero', data["CapitalGain"]!='high' ),"CapitalGain"] = 'low'

    data.loc[(data["CapitalLoss"] > 1874.19),"CapitalLoss"] = 'high'
    data.loc[(data["CapitalLoss"] == 0 ,"CapitalLoss")]= 'zero'
    data.loc[operator.and_(data["CapitalLoss"]!='zero', data["CapitalLoss"]!='high'),"CapitalLoss"] = 'low'


    #NativeCountry 41---> 2
def NativeCountry(data):
    
    datai = [data]

    for dataset in datai:
        dataset.loc[dataset["NativeCountry"] != ' United-States', "NativeCountry"] = 'Non-US'
        dataset.loc[dataset["NativeCountry"] == ' United-States', "NativeCountry"] = 'US'


# MaritalStatus  7 --->2
def MaritalStatus(data):
    
    data["MaritalStatus"] = data["MaritalStatus"].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')
    data["MaritalStatus"] = data["MaritalStatus"].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')

# Age 74 dimention   
# HoursPerWeek 96 dimention
def Discretization(data):
    data['Age']= pd.cut(data['Age'],bins=35)
    data['HoursPerWeek'] = pd.cut(data['HoursPerWeek'],bins=45)



def adult_preprocess(data):
    CapitalGainLoss(data)
    NativeCountry(data)
    MaritalStatus(data)
    discretizer = KBinsDiscretizer(n_bins=45, encode='ordinal', strategy='uniform')
    data['HoursPerWeek'] = discretizer.fit_transform(data['HoursPerWeek'].values.reshape(-1,1))
    #data['HoursPerWeek'] = pd.cut(data['HoursPerWeek'],bins=45,labels=False)
    con_vars = ['Age']
    cat_vars = [col for col in data.columns if col not in con_vars]
    return data, cat_vars
    
def load_adult():
    df_adult = pd.read_csv('./dataset/combined_set.csv')
    data, cat_vars = adult_preprocess(df_adult)
    data_info = get_data_info(data,cat_vars)
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=2)
    print("Data info:", data_info) 
    return train_data, test_data,data_info,cat_vars
