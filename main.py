import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

def load_image(file_path):
    image = Image.open(file_path)
    return image

def display_image(image):
    root = tk.Tk()
    img = ImageTk.PhotoImage(image)
    panel = tk.Label(root, image=img)
    panel.pack(side="bottom", fill="both", expand="yes")
    root.mainloop()

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def predict_image(model, image):
    image = image.resize((128, 128))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    return np.argmax(predictions)

def main():
    model_path = 'trained_plant_disease_model.keras'
    model = load_model(model_path)

    class_name = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ]

    while True:
        file_path = filedialog.askopenfilename()
        if not file_path:
            break

        image = load_image(file_path)
        display_image(image)

        result_index = predict_image(model, image)
        print(f"Model is predicting it's a {class_name[result_index]}")

        user_input = input("Do you want to choose another image? (yes/no): ")
        if user_input.lower() != 'yes':
            break

if __name__ == "__main__":
    main()