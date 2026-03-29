import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template_string
from PIL import Image
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Activation, Dropout
import base64
from io import BytesIO

app = Flask(__name__)


def build_model_skeleton():
    base = ResNet50V2(input_shape=(224, 224, 3), include_top=False, weights=None)
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(64),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.load_weights("ResNet50_Transfer_Learning.keras")
    return model

model = build_model_skeleton()
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def process_image_as_gray(img):
    
    gray_img = img.convert('L').convert('RGB') 
    gray_img = gray_img.resize((224, 224))
    img_array = np.array(gray_img) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- SIMPLE FRONTEND ---
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Emotion Detector</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background-color: #f4f4f4; padding: 50px; }
        .main-box { background: white; padding: 30px; border-radius: 10px; display: inline-block; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
        .result-box { margin-top: 30px; border-top: 2px solid #eee; padding-top: 20px; }
        img { border-radius: 10px; max-width: 300px; margin-bottom: 10px; }
        .emotion-name { font-size: 24px; color: #007bff; font-weight: bold; }
        input, button { margin: 10px; padding: 10px; cursor: pointer; }
        button { background-color: #007bff; color: white; border: none; border-radius: 5px; }
    </style>
</head>
<body>

    <div class="main-box">
        <h1>Emotion AI</h1>
        <p>Upload a face photo to predict emotion</p>
        
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <br>
            <button type="submit">Predict Emotion</button>
        </form>

        {% if emotion %}
        <div class="result-box">
            <h3>Original Image:</h3>
            <img src="data:image/png;base64,{{ img_str }}">
            <p>I think the emotion is:</p>
            <div class="emotion-name">{{ emotion }}</div>
            <p>Confidence: {{ (confidence * 100)|round(1) }}%</p>
            <a href="/">Try Another One</a>
        </div>
        {% endif %}
    </div>

</body>
</html>
'''

# --- BACKEND ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def home():
    emotion, confidence, img_str = None, None, None
    
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            
            original_img = Image.open(file).convert('RGB')
            
            
            ai_data = process_image_as_gray(original_img)
            
            # 3. Get Prediction
            preds = model.predict(ai_data)
            emotion = class_names[np.argmax(preds)]
            confidence = float(np.max(preds))
            
            # 4. Prepare original image for display
            buffered = BytesIO()
            original_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

    return render_template_string(HTML_TEMPLATE, emotion=emotion, confidence=confidence, img_str=img_str)

if __name__ == '__main__':
    app.run(debug=True)
