# 😊 Emotion Detection using Deep Learning

## 🚀 Overview

This project is a **Deep Learning-based Emotion Detection System** that predicts human emotions from facial images.

The model is built using **Transfer Learning (ResNet50V2)** and deployed using **Gradio on Hugging Face Spaces**.

---

## 🌐 Live Demo

👉 https://huggingface.co/spaces/aayansh26/emotion-detector

---

## 🧠 Model Repository

👉 https://huggingface.co/aayansh26/emotion_detection

---

## 🎯 Features

* 📷 Upload image to detect emotions
* 🧠 Deep Learning with Transfer Learning
* ⚡ Real-time prediction
* 🌐 Deployed on Hugging Face Spaces
* 📊 Top predictions with confidence scores

---

## 🧠 Model Architecture

* Base Model: **ResNet50V2**
* Custom Layers:

  * GlobalAveragePooling2D
  * BatchNormalization
  * Dense Layer (64 units)
  * Dropout (0.5)
  * Output Layer (7 classes, Softmax)

---

## 🎭 Emotion Classes

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

---

## 📊 Model Performance

* Achieved **75%** accuracy in traing & **69%** in testing  using transfer learning
* Performs well on real-world facial images

---

## ⚙️ How It Works

1. Upload an image
2. Image preprocessing:

   * Converted to grayscale
   * Resized to 224×224
   * Normalized
3. Model predicts emotion probabilities
4. Top predictions are displayed

---

## 🏗️ Project Structure

```
emotion_detection/
│
├── emotion_detection_training.ipynb   # Model training (Colab)
├── app.py                            # Gradio app
├── requirements.txt                  # Dependencies
├── README.md                         # Documentation

```

---

## ▶️ Run Locally

Clone the project:

```bash
git clone https://huggingface.co/spaces/aayansh26/emotion-detector
cd emotion-detector
pip install -r requirements.txt
python app.py
```

---

## ⚠️ Notes

* Model is hosted separately on Hugging Face
* Loaded dynamically using `hf_hub_download`
* Avoids large file upload in repository

---

## 👨‍💻 Author

**Aayansh Yadav**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

---
