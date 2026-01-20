import tensorflow as tf
import numpy as np
from PIL import Image
import io

class ModelService:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = (224, 224)

    def preprocess_image(self, image_bytes: bytes):
        # Open image and ensure it's RGB
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(self.img_size)
        
        # Convert to array and expand dimensions for model input
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # NOTE: If using MobileNetV2, rescaling is handled inside the model 
        # based on the script we wrote earlier.
        return img_array

    def predict(self, image_bytes: bytes):
        processed_img = self.preprocess_image(image_bytes)
        prediction = self.model.predict(processed_img)[0][0]
        
        label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        
        return {"result": label, "confidence": round(confidence, 4)}