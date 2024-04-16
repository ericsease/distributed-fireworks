import json
import os
import subprocess

import numpy as np
import psutil
import requests
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from joblib import load

app = Flask(__name__, static_folder='build', static_url_path='/')
CORS(app)


# Update to load a Naive Bayes model
def load_model(model_path):
    model = load(model_path)
    return model


model_path = os.path.join(os.path.dirname(__file__), 'naive_bayes_model.joblib')
model = load_model(model_path)


def preprocess_image(img_file):
    """Resize, convert to grayscale, and flatten the image."""
    try:
        img = Image.open(img_file).convert('L')  # Convert image to grayscale
        img = img.resize((64, 64))  # Resize image to 64x64 pixels
        img_array = np.array(img).flatten()  # Flatten the image
        return img_array
    except Exception as e:
        return None


@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Please provide an image for classification'}), 400

    img_file = request.files['image']
    if img_file.content_type not in ['image/jpeg', 'image/png', 'image/gif']:
        return jsonify({'error': 'File type not supported. Please upload an image.'}), 400

    img_array = preprocess_image(img_file)
    if img_array is None:
        return jsonify({
                           'error': 'Could not process the uploaded file. Make sure it is a valid image.'}), 400

    # Predict the class using the Naive Bayes model
    predicted = model.predict([img_array])[0]
    predicted_label = 'Fireworks' if predicted == 1 else 'Non-Fireworks'

    return jsonify({'classification': predicted_label})


def check_system_utilization():
    return {
        'cpu': psutil.cpu_percent(),
        'ram': psutil.virtual_memory().percent
    }


def select_peer():
    peers = load_peers()
    for peer in peers:
        try:
            response = requests.get(peer + '/system_utilization')
            if response.status_code == 200 and response.json()['cpu'] < 80:
                return peer
        except requests.RequestException:
            continue
    return None


def load_peers():
    with open('peers.json') as f:
        data = json.load(f)
    return data['peers']


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status="up"), 200


@app.route('/peers', methods=['GET'])
def list_peers():
    return jsonify(peers=load_peers()), 200


@app.route('/ping_peers', methods=['GET'])
def ping_peers():
    peers = load_peers()
    ping_results = {}
    for peer in peers:
        result = subprocess.run(["ping", "-c", 4, peer], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        ping_results[peer] = result.stdout
    return jsonify(ping_results=ping_results), 200


@app.route('/system_utilization', methods=['GET'])
def system_utilization():
    return jsonify(cpu_usage=psutil.cpu_percent(), ram_usage=psutil.virtual_memory().percent)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1717)
