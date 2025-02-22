import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from datetime import datetime, timezone
from pymongo import MongoClient
from werkzeug.utils import secure_filename

app = Flask(__name__)

# MongoDB setup
client = MongoClient("mongodb+srv://leaflens:leaflens@cluster0.gpg4e.mongodb.net/LeafLens?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true")
db = client.LeafLens
results_collection = db.model_prediction

# Folder to save uploaded images
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load models with memory management
first_model = tf.keras.models.load_model('riceleaf_identification.keras')
second_model = tf.keras.models.load_model('TAN_model1.h5')

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size):
    img = tf.keras.utils.load_img(img_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files or 'userId' not in request.form:
            return jsonify({"error": "Image or userId not provided"}), 400

        file = request.files['image']
        user_id = request.form['userId']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            img_array = preprocess_image(file_path, target_size=(224, 224))

            # Run predictions within a TensorFlow session to manage memory
            with tf.device('/cpu:0'):
                first_predictions = first_model.predict(img_array)
                first_predicted_class = float(first_predictions[0][0])

                if first_predicted_class < 0.5:
                    second_predictions = second_model.predict(img_array)
                    second_predicted_class = int(np.argmax(second_predictions, axis=-1)[0])

                    result = {
                        'userId': user_id,
                        'image_path': file_path,
                        'first_model_result': first_predicted_class,
                        'second_model_class': second_predicted_class,
                        'createdAt': datetime.now(timezone.utc)
                    }
                    results_collection.insert_one(result)

                    os.remove(file_path)
                    return jsonify({
                        "message": "Image analyzed successfully",
                        "first_model_result": first_predicted_class,
                        "second_model_class": second_predicted_class,
                    }), 200
                else:
                    os.remove(file_path)
                    return jsonify({"message": "The taken image is not a rice leaf. Please capture again."}), 400
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
