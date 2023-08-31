 
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

df = pd.read_csv("ADHD Data Set 2.csv")


import pandas as pd



def train_dummies(train_df):

    onehotencoder_dict = {}
    encoded_X_train = None
    for col in train_df.columns:

        if col not in ["Age" , "Grade"]:

            arr = train_df[col].values
            train_arr = arr.reshape(-1,1)
            col_enc = OneHotEncoder(handle_unknown='ignore')
            col_enc.fit(train_arr)
            onehotencoder_dict[col] = col_enc
            
            
    return onehotencoder_dict 
        
def test_encoding(data, enc_dict):
    output = pd.DataFrame({})
    for col in data.columns:
        if col  not in ["Age" , "Grade"]:
            enc = enc_dict[col] 
            arr = data[col].values
            arr = arr.reshape(-1,1)
            
            classes = [col + "_" + str(x) for x in enc.categories_[0]]
            feature = enc.transform(arr).toarray()

            features = pd.DataFrame(feature)
            features.columns = classes
            
            output = pd.concat([output , features],axis =1)
    return output

def label_encoder(target):
    labels = target.apply(lambda x : {"Yes":1 , "No":0}[x])
    
    return labels



