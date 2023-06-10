import tensorflow as tf
import numpy as np
import torch
from joblib import load
import cv2

from pathlib import Path
import torchvision.models as models

import tkinter as tk
from tkinter import Tk, Canvas, Entry, Text, Label, Button, PhotoImage, filedialog, ttk, messagebox
from PIL import Image, ImageTk


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\galse\DS50\interface\assets")

class_names = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

model_tensorflow = tf.keras.models.load_model('C:/Users/galse/DS50/models/models_train_alone', compile=False)
model_tensorflow.compile()

model_randomForest = load('C:/Users/galse/DS50/models/randomForest.joblib')

model_knn = load('C:/Users/galse/DS50/models/knn_model.joblib')


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def choose_model():
    if selected_model.get() == 'Pytorch' :
        image_path = openFileDialog()
        results = predict_pytorch(image_path)
        display_result(results)
        changeImageShown(image_path)
    elif selected_model.get() == 'Tensorflow' :
        image_path = openFileDialog()
        results = predict_tensorflow(image_path)
        display_result(results)
        changeImageShown(image_path)
    elif selected_model.get() == 'Random Forest' :
        image_path = openFileDialog()
        results = predict_randomForest(image_path)
        display_result(results)
        changeImageShown(image_path)
    elif selected_model.get() == 'KNN':
        image_path = openFileDialog()
        results = predict_knn(image_path)
        display_result(results)
        changeImageShown(image_path)
    elif selected_model.get() == 'Tous les modèles':
        image_path = openFileDialog()
        predict_all(image_path)
        changeImageShown(image_path)
    else:
        messagebox.showinfo("Information","Aucune modèle choisi. Veuillez en choisir un dans le menu déroulant")

def openFileDialog():
    return filedialog.askopenfilename(initialdir="/", title="Sélectionner une image",
                                            filetypes=(("Fichiers image", "*.jpg *.jpeg *.png"), ("Tous les fichiers", "*.*")))

def changeImageShown(img_path):
    print(img_path)
    max_width = 340
    img = Image.open(img_path)
    pixels_x, pixels_y = tuple([int(max_width/img.size[0] * x)  for x in img.size])
    new_img = ImageTk.PhotoImage(img.resize((pixels_x, pixels_y)))
    #new_img = PhotoImage(file=img_path)
    canvas.img_shown = new_img
    canvas.itemconfigure(image_1, image=new_img)

def display_result(results):
    print("display result")
    predicted_class_name = results["predicted_class_name"]
    predicted_class_probability = results["predicted_class_score"]
    result_label.config(text="Cette image appartient à la classe {} avec une probabilité à {}%."
                              .format(predicted_class_name, predicted_class_probability))

def predict_all(image_path):
    print("Prédictions en utilisant tous les modèles")
    result_knn = predict_knn(image_path)
    result_tensorflow = predict_tensorflow(image_path)
    result_randomForest = predict_randomForest(image_path)
    result_resnet = predict_pytorch(image_path)

    result_label.config(text="CNN :\t\t{} confiance de {}%\nResNet34 :\t\t{} confiance de {}%\nKNN :\t\t{} confiance de {}%\nRandom Forest :\t\t{} probabilité de {}%"
                        .format(
                            result_tensorflow["predicted_class_name"], result_tensorflow["predicted_class_score"],
                            result_resnet["predicted_class_name"], result_resnet["predicted_class_score"],
                            result_knn["predicted_class_name"], result_knn["predicted_class_score"],
                            result_randomForest["predicted_class_name"], result_randomForest["predicted_class_score"]
                        ))



def predict_pytorch(image_path):
    predicted_class_name = "pytorch"
    predicted_class_probability = 100

    results = {
        "predicted_class_name" : predicted_class_name,
        "predicted_class_score" : predicted_class_probability
    }
    return results

def predict_knn(image_path):
    image = cv2.imread(image_path)

    resized_image = cv2.resize(image, (435, 303))

    flattened_image = resized_image.reshape(-1)

    predicted_class_index = int(model_randomForest.predict([flattened_image]))
    predicted_class_name = class_names[predicted_class_index]
    predicted_probabilities = model_knn.predict_proba([flattened_image])
    predicted_class_probability = predicted_probabilities[0][predicted_class_index]
    
    results = {
        "predicted_class_name" : predicted_class_name,
        "predicted_class_score" : predicted_class_probability
    }
    return results

def predict_randomForest(image_path):
    image = cv2.imread(image_path)

    # Redimensionner l'image à la même taille que les images d'entraînement
    resized_image = cv2.resize(image, (435, 303))

    # Aplatissement de l'image en une seule dimension
    flattened_image = resized_image.reshape(-1)

    # Effectuer la prédiction
    predicted_class_index = int(model_randomForest.predict([flattened_image]))
    predicted_class_name = class_names[predicted_class_index]
    predicted_probabilities = model_randomForest.predict_proba([flattened_image])
    predicted_class_probability = predicted_probabilities[0][predicted_class_index]

    results = {
        "predicted_class_name" : predicted_class_name,
        "predicted_class_score" : predicted_class_probability
    }
    return results

def predict_tensorflow(image_path):
    # Ouvrir l'explorateur de fichiers pour sélectionner une image
    img = tf.keras.utils.load_img(
        image_path, target_size=(224, 224)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # Faire la prédiction
    predictions = model_tensorflow.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "Cette image appartient à la classe {} avec une confiance à {:.2f}%."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    results = {
        "predicted_class_name" : class_names[np.argmax(score)],
        "predicted_class_score" : 100 * np.max(score)
    }
    return results

window = Tk()

window.title("Prédicteur")

window.geometry("663x585")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 585,
    width = 663,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    663.0,
    585.0,
    fill="#9B6C6C",
    outline="")

image_image_1 = PhotoImage(
    file="C:/Users/galse/DS50/interface/assets/frame0/image_1.png")
image_1 = canvas.create_image(
    336.0,
    249.0,
    image=image_image_1
)

button_image_1 = PhotoImage(
    file=relative_to_assets("C:/Users/galse/DS50/interface/assets/frame0/button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=choose_model,
    relief="flat"
)
button_1.place(
    x=212.0,
    y=391.0,
    width=238.0,
    height=60.0
)

canvas.create_text(
    191.0,
    61.0,
    anchor="nw",
    text="Détecteur de cancer",
    fill="#FFFFFF",
    font=("Inter", 30 * -1)
)

result_label = Label(
    canvas,
    anchor="nw",
    text="",
    font=("Inter", 15 * -1),
    bg="#9B6D6D"
)
result_label.place(
    x=64.0,
    y=508.0
)

# create a combobox
selected_model = tk.StringVar()
selected_model_cb = ttk.Combobox(window, textvariable=selected_model)
selected_model_cb['values'] = ['Sélectionnez modèle','Pytorch', 'Tensorflow', 'Random Forest', 'KNN', 'Tous les modèles']
selected_model_cb['state'] = 'readonly'
selected_model_cb.place(x=64.0, y=475)


window.resizable(False, False)
window.mainloop()
