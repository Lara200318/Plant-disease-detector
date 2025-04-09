import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = load_model("model.h5")

labels = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Septoria Leaf Spot',
          'Spider Mites', 'Target Spot', 'Yellow Leaf Curl Virus', 'Mosaic Virus', 'Healthy']

def predict(img):
    img = img.resize((256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)[0]
    label = labels[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)
    return f"{label} ({confidence}%)"

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs="text",
                    title="Plant Disease Detector 🌿",
                    description="Upload a plant leaf image to identify disease.")

demo.launch()
