import io
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def predict_image(model, file, target_size=(128, 128), categories=None):
    
    # Convert FileStorage to BytesIO
    img_bytes = file.read()
    img = io.BytesIO(img_bytes)
    
    # Load image
    image = load_img(img, target_size=target_size)
    
    # Preprocess image
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    
    return categories[predicted_class] if categories else predicted_class
