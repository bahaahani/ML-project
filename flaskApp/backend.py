from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = load_model('./natural_images_model.h5')  # Load your trained model
class_names = ['airplane', 'car', 'cat', 'dog',
               'flower', 'fruit', 'motorbike', 'person']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    image = image.resize((64, 64))
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    # Get the index of the highest probability to find the predicted class
    predicted_class = class_names[np.argmax(prediction)]

    return jsonify({'prediction': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)
