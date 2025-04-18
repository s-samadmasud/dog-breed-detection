from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Create an app
app = Flask(__name__)

# Define your model path
model_path = 'models/fine_tuned_inception.h5'  # Update with your model location

# Define class labels (modify if needed)
class_labels = ['beagle', 'bulldog', 'dalmatian', 'german-shepherd', 'husky', 'labrador-retriever', 'poodle', 'rottweiler']

# Function to preprocess and predict
def predict_and_display(img):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence_level = np.max(predictions)
    predicted_class_name = class_labels[predicted_class]

    return predicted_class_name, confidence_level

# Route for image upload (API endpoint)
#@app.route('/')
#def index():
    return redirect(url_for('predict'))

# Route for the homepage (navbar)
@app.route('/')
def home():
    return redirect(url_for('predict'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Handle GET request (e.g., display form)
    if request.method == 'GET':
        return render_template('index.html')  # Assuming you have an index.html template
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('static', filename)
        file.save(filepath)
        img = image.load_img(filepath, target_size=(224, 224))

        predicted_class_name, confidence_level = predict_and_display(img)

        # Create a response string with the desired format
        response_text = f"Predicted Class: {predicted_class_name}, True Class: {predicted_class_name}"
    # Handle POST request (prediction logic)
    if request.method == 'POST':
        # ... (your prediction code using the uploaded image)
        return render_template('result.html', response_text=predicted_class_name, confidence=confidence_level,filename=filename)  # Assuming you have a result.html template
    else:
        return jsonify({'message': 'Invalid file format'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')