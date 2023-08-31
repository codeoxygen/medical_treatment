from flask import Flask, request, jsonify
from preprocessing import  test_encoding
import pandas as pd
import numpy as np
import train
import pickle

app = Flask(__name__)

# Load the model from the pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        df = pd.DataFrame({})
        data = request.json

        for col in train.column_sequence:
            value = data[col]
            arr = np.asarray([[value]])


        # Make predictions using the loaded model
        prediction = model.predict([features])

        # You can return the prediction as JSON
        response = {'prediction': prediction[0]}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
