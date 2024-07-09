import numpy as np # type: ignore
from flask import Flask, request, jsonify# type: ignore
from tensorflow.keras.models import load_model# type: ignore
import pandas as pd# type: ignore
from flask_cors import CORS# type: ignore

# Load the model
model = load_model('crop_recommendation_model.h5')

# Load the label categories
df = pd.read_csv('Crop_recommendation.csv')
df['label'] = pd.Categorical(df['label'])
categories = df['label'].cat.categories

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

@app.route('/')
def start():
    return 'Server is running'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Ensure all input values are converted to float
        conditions = np.array([
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['pH']),
            float(data['rainfall'])
        ]).reshape(1, 7)

        prediction = model.predict(conditions)
        predicted_class = np.argmax(prediction, axis=1)
        crop = categories[predicted_class[0]]
        return jsonify({'recommended_crop': crop})
    except ValueError as ve:
        return jsonify({'error': 'Invalid input data format: ' + str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

