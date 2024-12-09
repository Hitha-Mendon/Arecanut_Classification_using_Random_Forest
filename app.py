from flask import Flask, request, render_template
import joblib
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('arecanut_model.pkl')
IMAGE_SIZE = (64, 64)

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    try:
        # Process the image using OpenCV
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode image
        img_resized = cv2.resize(img, IMAGE_SIZE).flatten()  # Resize and flatten
        prediction = model.predict([img_resized])[0]
        labels = ['High', 'Medium', 'Low']
        result = labels[prediction]

        return render_template('result.html', result=result)
    except Exception as e:
        return f"Error processing file: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
