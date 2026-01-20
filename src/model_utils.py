import tensorflow as tf
import numpy as np
from PIL import Image

def load_my_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_image(model, image):
    # Preprocessing
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Inference
    prediction = model.predict(img_array)
    probability = float(prediction[0][0])
    
    # Logic
    result = "PNEUMONIA" if probability > 0.5 else "NORMAL"
    confidence = probability if result == "PNEUMONIA" else 1 - probability
    
    return result, confidence