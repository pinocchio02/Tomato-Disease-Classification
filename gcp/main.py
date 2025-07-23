from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
from flask import jsonify  # IMPORTANT: Cloud Functions expects Flask-style response
import functions_framework  # REQUIRED for GCP to detect this as an HTTP function
from io import BytesIO
from flask import Request  # optional, for type hinting

model = None
CLASS_NAMES = [
    'Bacterial Spot',
    'Early Blight',
    'Late Blight',
    'Leaf Mold',
    'Septoria Leaf Spot',
    'Spider Mites_Two Spotted Spider Mite',
    'Target Spot',
    'YellowLeaf Curl Virus',
    'Mosaic Virus',
    'Healthy'
]

BUCKET_NAME = "pinocchio4tayyy"

def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}.")

# ✅ This decorator tells Google Cloud Functions this is the HTTP handler
@functions_framework.http

def predict(request):  # type: (Request) -> dict
    global model

    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/clean_model.h5",
            "/tmp/clean_model.h5",
        )
        model = tf.keras.models.load_model("/tmp/clean_model.h5")

    # ✅ Fix: Read raw image bytes correctly from GCP request
    if request.method != 'POST':
        return {"error": "Only POST requests are allowed."}, 405

    if not request.files and not request.data:
        return {"error": "No image file found in request."}, 400

    try:
        # If using multipart/form-data (Postman file upload), use this:
        file = request.files.get("file")
        if file is None:
            return {"error": "File not found in request."}, 400

        # ✅ Read image and preprocess
        image = Image.open(file).convert("RGB").resize((160, 160))
        image = np.array(image) / 255.0
        img_array = tf.expand_dims(image, 0)

        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = round(100 * np.max(predictions[0]), 2)

        return {"class": predicted_class, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}, 500


# def predict(request):
#     global model
#     if model is None:
#         download_blob(BUCKET_NAME, "models/clean_model.h5", "/tmp/clean_model.h5")
#         model = tf.keras.models.load_model("/tmp/clean_model.h5")

#     if request.method != "POST":
#         return jsonify({"error": "Only POST method is supported."}), 405

#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     try:
#         image.open(image).convert("RGB").resize((160, 160))
#         image = np.array(image) / 255.0
#         img_array = np.expand_dims(image, axis=0)
#         predictions = model.predict(img_array)
#         predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#         confidence = round(100 * np.max(predictions[0]), 2)

#         return jsonify({
#             "class": predicted_class,
#             "confidence": confidence
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


