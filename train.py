import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing import label_encoder , train_dummies , test_encoding
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import pickle
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv('ADHD Data Set 2.csv')
df.drop(columns=["Timestamp", "Name"], inplace=True)
df = df.dropna()
columns_to_drop = ['Diagnosis']
X = df.drop(columns_to_drop, axis=1)
y = df['Diagnosis']

y = label_encoder(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
column_sequence  = X_train.columns

enc_dict = train_dummies(X_train)

encoded_train = test_encoding(X_train, enc_dict)
encoded_test = test_encoding(X_test, enc_dict)


clf = RandomForestClassifier()
clf.fit(encoded_train,y_train)
y_pred = clf.predict(encoded_test)

print(f'Test accuracy score : {accuracy_score(y_test , y_pred)}')

joblib.dump(clf, 'model.pkl')

    

