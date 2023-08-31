# -*- coding: utf-8 -*-
"""Random Forest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vVrFMYHeXw8cl_ySnRyz7jfAghtsn8sv
"""

from google.colab import files
uploaded = files.upload()
import pickle

import pandas as pd

df = pd.read_csv('ADHD Data Set 2.csv')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Read the dataset
df = pd.read_csv('ADHD Data Set 2.csv')

# Identify the features (X) and the target variable (y)
columns_to_drop = ['Diagnosis']
X = df.drop(columns_to_drop, axis=1)
y = df['Diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=150, random_state=42)

# Perform one-hot encoding on X_train and X_test
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

# Realign the columns in X_test_encoded to match X_train_encoded
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Encode the target variable
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train the classifier on the encoded training data
random_forest.fit(X_train_encoded, y_train_encoded)

import pickle
import joblib
model_name = "model.pkl"
joblib.dump(random_forest, model_name)

# Make predictions on the testing data
y_pred_encoded = random_forest.predict(X_test_encoded)

# Decode the predicted labels back to their original form
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)