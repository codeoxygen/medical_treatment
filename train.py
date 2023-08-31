import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing import dummie_encoding
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv('ADHD Data Set 2.csv')
df.drop(columns=["Timestamp", "Name"], inplace=True)

columns_to_drop = ['Diagnosis']
X = df.drop(columns_to_drop, axis=1)
y = df['Diagnosis']


labelencoder_dict = {}
onehotencoder_dict = {}
encoded_x = None
column_names = []


for i in range(0, X.shape[1]):

    feature = X.iloc[:, i].values
    feature = feature.reshape(X.shape[0], 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    feature = onehot_encoder.fit_transform(feature)
    onehotencoder_dict[i] = onehot_encoder
    column_names.extend(onehot_encoder.get_feature_names_out(X.columns[i]))
    if encoded_x is None:
      encoded_x = feature
    else:
      encoded_x = np.concatenate((encoded_x, feature), axis=1)

X_train, X_test, y_train, y_test = train_test_split(encoded_x, y, test_size=0.2, random_state=42)

encoded_df = pd.DataFrame(encoded_x, columns=column_names)
print(encoded_df.head())



#X_train, X_test, y_train, y_test = train_test_split(encoded_x, y, test_size=0.2, random_state=42)