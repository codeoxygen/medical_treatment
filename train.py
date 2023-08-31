import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing import get_dummies , train_dummies
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

non_cat_col = ["Age" , "Grade"]
print(X_test.head(10))
X_train = X_train.drop(columns=non_cat_col,axis=1)
X_test = X_test.drop(columns=non_cat_col,axis=1)

ohedict , train = train_dummies(cat_df=X_train,raw_data=X_train)

_, test = train_dummies(cat_df=X_train,raw_data=X_test)
