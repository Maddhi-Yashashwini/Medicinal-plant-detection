from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image
import requests

# === App Setup ===
app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)

# === Load Model ===
model = load_model('model/xception_plant_model.h5')

# === Load Class Names ===
def load_class_names():
    try:
        with open('class_names.txt', 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error loading class names: {e}")
        return []

class_names = load_class_names()

# === upload folder ===
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Gemini API key here
GEMINI_API_KEY = "AIzaSyA2YvuCDjuDoXFHg5bTYtviImsQAKQvBnI" 
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

def get_plant_use_case(plant_name):
    prompt = f"What are the medicinal uses of the {plant_name} plant, give in short 5 sentences?"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        # Extract the generated text from the Gemini API response
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        return "Use case information not available."

# === Serve Frontend HTML ===
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# === Prediction Endpoint ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Preprocess image for Xception
        img = Image.open(file_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        use_case = get_plant_use_case(predicted_class)

        return jsonify({'prediction': predicted_class, 'use_case': use_case})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True, port=5000)
