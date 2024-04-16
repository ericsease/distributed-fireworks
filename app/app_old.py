import json
import os
import subprocess
from flask_cors import CORS

import psutil
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from torchvision import models, transforms


app = Flask(__name__, static_folder='build', static_url_path='/')
CORS(app)


# Define the model architecture to match your trained model
def initialize_model(num_classes=2):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# Load the trained model
def load_model(model_path):
    model = initialize_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


model_path = os.path.join(os.path.dirname(__file__), 'firework_classifier.pth')
model = load_model(model_path)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.route('/')
def serve():
    # Serve the index.html file from the static folder
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/classify', methods=['POST'])
def classify_image():
    # Class names must match those used during training
    class_names = ['Fireworks', 'Non-Fireworks']

    allowed_mime_types = ['image/jpeg', 'image/png', 'image/gif']

    if 'image' not in request.files:
        return jsonify({'error': 'Please provide an image for classification'}), 400

    img_file = request.files['image']

    # Validate MIME type
    if img_file.content_type not in allowed_mime_types:
        return jsonify({'error': 'File type not supported. Please upload an image.'}), 400

    try:
        img = Image.open(img_file.stream).convert('RGB')
    except Exception as e:
        return jsonify({
            'error': 'Could not process the uploaded file. Make sure it is a valid image.'}), 400

    # Apply the same transformations as during training
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        predicted_label = class_names[predicted.item()]

    return jsonify({'classification': predicted_label})


def load_peers():
    use_docker = os.getenv('USE_DOCKER_SERVICE_NAMES', 'false').lower() == 'true'
    file_name = 'peers_docker.json' if use_docker else 'peers.json'
    with open(file_name) as f:
        data = json.load(f)
    return data['peers']


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status="up"), 200


@app.route('/peers', methods=['GET'])
def list_peers():
    peers = load_peers()
    return jsonify(peers=peers), 200


@app.route('/ping_peers', methods=['GET'])
def ping_peers():
    peers = load_peers()
    ping_results = {}
    for peer in peers:
        # Ping each peer, sending 4 packets ("-c 4")
        result = subprocess.run(["ping", "-c", "4", peer], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        ping_results[peer] = result.stdout
    return jsonify(ping_results=ping_results), 200


@app.route('/system_utilization', methods=['GET'])
def system_utilization():
    cpu_usage = psutil.cpu_percent()  # CPU usage as a percentage
    ram_usage = psutil.virtual_memory().percent  # RAM usage as a percentage

    return jsonify(cpu_usage=cpu_usage, ram_usage=ram_usage)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1717)
