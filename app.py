import os
import gdown
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from predict import predict_image

app = Flask(__name__)

MODEL_PATH = "skin_disease_model.h5"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = os.environ.get("MODEL_URL")
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
model = load_model(MODEL_PATH)

# Categories of diseases
categories = ['eczema', 'melanoma', 'atopic_dermatitis', 'basal_cell_carcinoma', 'melanotic_nevi']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No file selected", 400
    
    # Pass MODEL instead of path
    result = predict_image(model, file, categories=categories)
    
    return render_template('detect.html', prediction=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
