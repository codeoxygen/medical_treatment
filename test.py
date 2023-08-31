# Example usage\


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("ADHD Data Set 2.csv")


columns_to_drop = ['Diagnosis']
X = df.drop(columns_to_drop, axis=1)
y = df['Diagnosis']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train = X_train['AH-INA1']
test = X_test["AH-INA1"]
print(train.head(10))
arr = train.values
train_arr= arr.reshape(-1,1)
#print(train_arr)

test_arr = test.values.reshape(-1,1)
#print(f'before reshape {arr} \nAfter reshape : {arr.reshape(1,-1)}')

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(train_arr)
print(enc.categories_)
print(list(enc.categories_))

data = enc.transform(train_arr).toarray()[:10]
columns = ["AH-INA1"+"_"+col for col in enc.categories_]
print(pd.DataFrame(data,columns = columns))


print("----------------TEST----------------------")

