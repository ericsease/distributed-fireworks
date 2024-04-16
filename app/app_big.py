import json
import os
import subprocess

import psutil
import requests
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
from torchvision import models, transforms

app = Flask(__name__, static_folder='build', static_url_path='/')
CORS(app)


def initialize_model(num_classes=2):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def load_model(model_path):
    model = initialize_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


model_path = os.path.join(os.path.dirname(__file__), 'firework_classifier.pth')
model = load_model(model_path)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Please provide an image for classification'}), 400

    # Check system utilization before processing
    if check_system_utilization()['cpu'] > 80:  # Assume threshold is 80% CPU usage
        peer = select_peer()
        if peer:
            return redirect(peer + '/classify', code=307)

    img_file = request.files['image']
    if img_file.content_type not in ['image/jpeg', 'image/png', 'image/gif']:
        return jsonify({'error': 'File type not supported. Please upload an image.'}), 400

    try:
        img = Image.open(img_file.stream).convert('RGB')
    except Exception as e:
        return jsonify({
            'error': 'Could not process the uploaded file. Make sure it is a valid image.'}), 400

    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        predicted_label = ['Fireworks', 'Non-Fireworks'][predicted.item()]

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
