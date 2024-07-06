import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

# Path to the TFLite model
model_export = 'best_float32.tflite'

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_export)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to load and preprocess image
def load_image(image, target_size):
    img = Image.open(image).convert('RGB').resize(target_size)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0  # Normalize if needed
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Load and preprocess the image
    input_shape = input_details[0]['shape'][1:3]
    image = load_image(file, input_shape)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get the output data
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Load the label map into memory
    with open("labels.txt", 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]

    # Get bounding box coordinates from output data
    boxes = output_data[0]

    # Format the results
    threshold = 0.5  # Minimum probability threshold

    # Get the index of the highest probability
    best_index = np.argmax(boxes)
    best_prob = boxes[best_index]

    if best_prob > threshold:
        ymin, xmin, ymax, xmax = 0, 0, 1, 1  # Assuming whole image for simplicity
        best_result = {
            "isDetected": True,
            'label': labels[best_index],
            'probability': float(best_prob),  # Convert to standard float
            'bounding_box': [float(ymin), float(xmin), float(ymax), float(xmax)]  # Convert to standard floats
        }
        return jsonify(best_result), 200
    else:
        return jsonify(
            {"isDetected": False, 'label': None, 'probability': None, 'bounding_box': None}
        ), 200

if __name__ == '__main__':
    app.run(debug=True)
