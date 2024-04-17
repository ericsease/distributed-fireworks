import json
import os

import numpy as np
import psutil
import requests
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from joblib import load

app = Flask(__name__, static_folder='build', static_url_path='/')
CORS(app)


# Update to load a Random Forest Model
def load_model(model_path):
    model = load(model_path)
    return model


model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model.joblib')
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


### Load balancing functions
def check_local_overload():
    util = check_system_utilization()
    return util['cpu'] > 80 or util['ram'] > 80 or util['load_1m'] > os.cpu_count() or util[
        'iowait'] > 20


def forward_request(peer, file_storage):
    files = {'image': (file_storage.filename, file_storage.read(), file_storage.content_type)}
    file_storage.seek(0)  # Reset file pointer to the start after reading
    url = f'http://{peer}:1717/classify'
    try:
        response = requests.post(url, files=files)
        return jsonify(response.json()), response.status_code
    except requests.RequestException as e:
        return jsonify({'error': 'Failed to forward request: ' + str(e)}), 500


### Routes

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Please provide an image for classification'}), 400

    img_file = request.files['image']

    if check_local_overload():
        peer = select_peer()
        if peer:
            return forward_request(peer, img_file)  # Pass only the img_file
        else:
            return jsonify(
                {'error': 'All systems are currently busy. Please try again later.'}), 503

    img_file = request.files['image']
    if img_file.content_type not in ['image/jpeg', 'image/png', 'image/gif']:
        return jsonify({'error': 'File type not supported. Please upload an image.'}), 400

    img_array = preprocess_image(img_file)
    if img_array is None:
        return jsonify({
            'error': 'Could not process the uploaded file. Make sure it is a valid image.'}), 400

    # Predict the class using the Naive Bayes model
    predicted = model.predict([img_array])[0]
    print(predicted)
    predicted_label = 'Fireworks' if predicted == 1 else 'Non-Fireworks'

    return jsonify({'classification': predicted_label})


def check_system_utilization():
    cpu = psutil.cpu_times_percent(interval=1)  # Collect CPU stats over 1 second
    load_avg = os.getloadavg()  # Get average system load over 1, 5, and 15 minutes
    io_wait = cpu.iowait  # Percent of time spent waiting for I/O

    return {
        'cpu': cpu.system + cpu.user,  # System and user CPU time
        'iowait': io_wait,
        'load_1m': load_avg[0],  # 1-minute load average
        'ram': psutil.virtual_memory().percent
    }


def select_peer():
    peers = load_peers()
    timeout = (5, 10)  # 5 seconds to connect, 10 seconds to read

    for peer in peers:
        url = f'http://{peer}:1717/system_utilization'
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                if (data['cpu'] < 80 and
                        data['ram'] < 80 and
                        data['load_1m'] < os.cpu_count() and
                        data['iowait'] < 20):
                    return peer
        except requests.RequestException as e:
            print(f"Failed to get response from {peer}, error: {str(e)}")
            continue
    return None  # Return None if no available peer found


def load_peers():
    with open('peers.json', 'r') as f:
        data = json.load(f)
    return data['peers']


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status="up"), 200


@app.route('/peers', methods=['GET'])
def list_peers():
    return jsonify(peers=load_peers()), 200


@app.route('/test-forward', methods=['POST'])
def test_forward():
    # Check if the part of the request containing the files has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    peer = "http://192.168.1.23:1717/classify"

    try:
        # Prepare the file payload for HTTP POST
        files = {'image': (file.filename, file.read(), file.content_type)}
        response = requests.post(peer, files=files, timeout=(5, 10))
        return jsonify(response.json()), response.status_code
    except requests.RequestException as e:
        return jsonify({'error': str(e)}), 500


@app.route('/system_utilization', methods=['GET'])
def system_utilization():
    return check_system_utilization(), 200


@app.route('/test-crash')
def test_crash():
    raise RuntimeError("Simulating a crash")  # This is uncaught and should terminate Flask.


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1717)
