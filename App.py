import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import uvicorn

# 1. API Initialize karna
app = FastAPI(title="AI Image Classifier API")

# 2. Pre-trained AI Model Load karna (MobileNetV2)
# Ye model 1000+ objects ko pehchan sakta hai
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def prepare_image(image_bytes):
    """Image ko AI model ke liye ready karne ka function"""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224)) # Model ko 224x224 size chahiye
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)

@app.get("/")
def home():
    return {"status": "Online", "message": "Welcome to my AI + DevOps Project!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """API endpoint jo image lega aur result dega"""
    # Image read karna
    image_bytes = await file.read()
    processed_image = prepare_image(image_bytes)
    
    # AI Prediction
    predictions = model.predict(processed_image)
    results = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    
    # Result return karna
    # results[0][0][1] = Object Name, [2] = Confidence score
    return {
        "prediction": results[0][0][1],
        "confidence": f"{results[0][0][2]*100:.2f}%"
    }

if __name__ == "__main__":
    # Server ko 8000 port par chalao
    uvicorn.run(app, host="0.0.0.0", port=8000)
