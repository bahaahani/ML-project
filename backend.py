from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = load_model('./natural_images_model.h5')  # Load your trained model


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    # Adjust size to match model's expected input
    image = image.resize((64, 64))
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    # Process the prediction and send back a human-readable result
    # This part depends on your model and what you're predicting

    return jsonify({'prediction': str(prediction)})


if __name__ == '__main__':
    app.run(debug=True)
