 
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

df = pd.read_csv("ADHD Data Set 2.csv")


def dummie_encoding(data):
    onehot_encoder = OneHotEncoder(sparse=False)
    transformed_data = onehot_encoder.fit_transform(data)

    return transformed_data

def label_Label_Encoder(data):
    label_en= data.customer_last_feedback.apply(lambda x: {"bad" :1, "good":2}[x])
    return data



