from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
from datetime import datetime, timezone
from pymongo import MongoClient
from werkzeug.utils import secure_filename

app = Flask(__name__)

# MongoDB setup
client = MongoClient("mongodb+srv://leaflens:leaflens@cluster0.gpg4e.mongodb.net/LeafLens?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true")
db = client.LeafLens
results_collection = db.model_prediction

# Folder to save the uploaded images
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load pre-trained models
first_model = tf.keras.models.load_model('riceleaf_identification.keras')  # Binary classification model
second_model = tf.keras.models.load_model('ColorClassification.h5')  # Multi-class classification model

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper function to check if file is allowed 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to preprocess the image
def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files or 'userId' not in request.form:
        return jsonify({"error": "Image or userId not provided"}), 400

    file = request.files['image']
    user_id = request.form['userId']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Ensure the uploads directory exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(file_path)  # Save file to uploads folder
        except Exception as e:
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

        try:
            # Preprocess the image
            img_array = preprocess_image(file_path, target_size=(224, 224))

            # First model prediction (Binary classification)
            first_predictions = first_model.predict(img_array)
            first_predicted_class = float(first_predictions[0][0])  # Unchanged logic from your code

            if first_predicted_class > 0.5:  # Only proceed if it is a rice leaf
                # Second model prediction (Multi-class classification)
                second_predictions = second_model.predict(img_array)
                second_predicted_class = int(np.argmax(second_predictions, axis=-1)[0])  # Class index

                # Save results to MongoDB
                result = {
                    'userId': user_id,
                    'image_path': file_path,
                    'first_model_result': first_predicted_class,  # Save first model result
                    'second_model_class': second_predicted_class,
                    'createdAt': datetime.now(timezone.utc)  # Timezone-aware datetime
                }
                results_collection.insert_one(result)

                # Clean up the uploaded file
                os.remove(file_path)

                return jsonify({
                    "message": "Image analyzed successfully",
                    "first_model_result": first_predicted_class,
                    "second_model_class": second_predicted_class,
                }), 200
            else:
                # Clean up the uploaded file if not a rice leaf
                os.remove(file_path)
                return jsonify({"message": "The taken image is not a rice leaf. Please capture again."}), 400

        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)  # Ensure cleanup even if an error occurs
            return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=port)
