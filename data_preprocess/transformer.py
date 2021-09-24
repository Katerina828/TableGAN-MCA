import numpy as np
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder

class BaseTransformer():
    
    def __init__(self,df,data_info):
        

        self.con_vars = [col[2] for col in data_info if col[1]=="tanh"]
        self.cat_vars = [col[2] for col in data_info if col[1]=="softmax"]

        self.columns_name = self.con_vars + self.cat_vars
        self.data = df[self.columns_name]
        
        self.con_loc =  [self.data.columns.get_loc(var) for var in self.con_vars]


    def transform(self):

        self.scaler = MinMaxScaler()
        self.enc = OneHotEncoder()
        con_columns = self.scaler.fit_transform(self.data[self.con_vars])
        cat_columns = self.enc.fit_transform(self.data[self.cat_vars]).toarray()
        data_np = np.column_stack((con_columns,cat_columns))

        return data_np

    def inverse_transform(self,data):

        data_con = self.scaler.inverse_transform(data[:,self.con_loc])
        data_cat = self.enc.inverse_transform(data[:,len(self.con_loc):])       
        data_inverse = np.column_stack((data_con,data_cat))
        print("Inverse transform completed!")
        return data_inverse