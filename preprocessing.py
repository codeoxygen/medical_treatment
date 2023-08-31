 
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

df = pd.read_csv("ADHD Data Set 2.csv")


import pandas as pd

def get_dummies(train):

    arr = train.values
    train_arr= arr.reshape(-1,1)
 
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_arr)
    return enc

def train_dummies(cat_df,raw_data):
    onehotencoder_dict = {}
    encoded_X_train = None
    for col in cat_df.columns:
        arr = cat_df[col].values
        train_arr = arr.reshape(-1,1)

        test_arr = raw_data[col].values
        test_arr = test_arr.reshape(-1,1)


        col_enc = OneHotEncoder(handle_unknown='ignore')
        col_enc.fit(train_arr)
        onehotencoder_dict[col] = col_enc
        data = col_enc.transform(test_arr).toarray()
        print(col_enc.categories_)
        columns = [ col + "_" + str(x) for  x in col_enc.categories_]
        
        features = pd.DataFrame(data=data)
        print(data)
        if encoded_X_train is None:
            encoded_X_train = features
        else:
            encoded_X_train = pd.concat([encoded_X_train,features],axis = 1)
    return onehotencoder_dict , encoded_X_train
        



def label_encoder(data):
    label_en= data.customer_last_feedback.apply(lambda x: {"bad" :1, "good":2}[x])
    return data



