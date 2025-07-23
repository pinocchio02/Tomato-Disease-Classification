from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# print(tf.__version__)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = r"C:/Users/raman/Deploy/Tomato Disease2/saved_models/clean_model"
MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['Bacterial Spot',
 'Early Blight',
 'Late Blight',
 'Leaf Mold',
 'Septoria Leaf Spot',
 'Spider Mites_Two Spotted Spider Mite',
 'Target Spot',
 'YellowLeaf Curl Virus',
 'Mosaic Virus',
 'Healthy']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((160, 160))  # ✅ Resize to model input
    image = np.array(image) / 255.0   # ✅ Normalize
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)