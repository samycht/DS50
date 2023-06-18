from tensorflow import keras
import numpy as np
import torch
from joblib import load
import cv2
from models_classes.googleNet import GoogleNetDetectionModel
from models_classes.denseNet import DenseNetDetectionModel
from models_classes.efficientNet import EfficientNetDetectionModel
from models_classes.resNext50 import ResNext50DetectionModel
from models_classes.resNet34 import ResNet34DetectionModel

import tkinter as tk
from tkinter import Tk, Canvas, Label, Button, PhotoImage, filedialog, ttk, messagebox
from PIL import Image, ImageTk

class_names = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

#Chargement des différents modèles
model_googleNet = GoogleNetDetectionModel()
googleNet_checkpoint = torch.load('C:/Users/galse/DS50_SAMBA/models/GoogleNet.pt')
model_googleNet.load_state_dict(googleNet_checkpoint)
model_googleNet.eval()
print("======\tModèle GoogleNet chargé")
model_denseNet = DenseNetDetectionModel()
denseNet_checkpoint = torch.load('C:/Users/galse/DS50_SAMBA/models/DenseNet.pt')
model_denseNet.load_state_dict(denseNet_checkpoint)
model_denseNet.eval()
print("======\tModèle DenseNet chargé")
model_efficientNet = EfficientNetDetectionModel()
efficientNet_checkpoint = torch.load('C:/Users/galse/DS50_SAMBA/models/EfficientNet.pt')
model_efficientNet.load_state_dict(efficientNet_checkpoint)
model_efficientNet.eval()
print("======\tModèle EfficientNet chargé")
model_resNext50 = ResNext50DetectionModel()
resNext50_checkpoint = torch.load('C:/Users/galse/DS50_SAMBA/models/ResNext50.pt')
model_resNext50.load_state_dict(resNext50_checkpoint)
model_resNext50.eval()
print("======\tModèle ResNext50 chargé")
model_resNet34 = ResNet34DetectionModel()
resNet34_checkpoint = torch.load('C:/Users/galse/DS50_SAMBA/models/ResNet34.pt')
model_resNet34.load_state_dict(resNet34_checkpoint)
model_resNet34.eval()
print("======\tModèle ResNet34 chargé")
model_randomForest = load('C:/Users/galse/DS50_SAMBA/models/randomForest.joblib')
print("======\tModèle Random Forest chargé")
model_knn = load('C:/Users/galse/DS50_SAMBA/models/knn.joblib')
print("======\tModèle KNN chargé")
model_XGBoost = load('C:/Users/galse/DS50_SAMBA/models/XGBoost.joblib')
print("======\tModèle XGBoost chargé")
model_KMeans = load('C:/Users/galse/DS50_SAMBA/models/k-means.joblib')
print("======\tModèle K-Means chargé")
model_logisticRegression = load('C:/Users/galse/DS50_SAMBA/models/logisticRegression.joblib')
print("======\tModèle de Regression logistique chargé")
model_vgg16 = keras.models.load_model('C:/Users/galse/DS50_SAMBA/models/vgg16.h5')
print("======\tModèle vgg16 chargé")


def choose_model():
    result_label_text.set("")
    if selected_model.get() == "Sélectionnez un modèle":
        messagebox.showinfo("Information","Aucune modèle choisi. Veuillez en choisir un dans le menu déroulant")
    else:
        image_path = openFileDialog()
        if selected_model.get() == 'Tous les modèles':
            predict_all(image_path)
        else:
            results = []
            if selected_model.get() == 'DenseNet' :
                results = predict_pytorch(image_path, "denseNet")
            elif selected_model.get() == 'EfficientNet' :
                results = predict_pytorch(image_path, "efficientNet")
            elif selected_model.get() == 'GoogleNet' :
                results = predict_pytorch(image_path, "googleNet")
            elif selected_model.get() == 'K-Means' :
                results = predict_machine_learning(image_path, "k-means")
            elif selected_model.get() == 'KNN':
                results = predict_machine_learning(image_path, "KNN")
            elif selected_model.get() == 'Random Forest' :
                results = predict_machine_learning(image_path, "randomForest")
            elif selected_model.get() == 'Régression logistique':
                results = predict_machine_learning(image_path, "logisticRegression")
            elif selected_model.get() == 'ResNet34' :
                results = predict_pytorch(image_path, "resNet34")
            elif selected_model.get() == 'ResNext50' :
                results = predict_pytorch(image_path, "resNext50")
            elif selected_model.get() == "VGG16":
                results = predict_keras(image_path)
            elif selected_model.get() == 'XGBoost':
                results = predict_machine_learning(image_path, "XGBoost")
            else:
                print("Choix du modèle non traité.")
            display_result(results)
        changeImageShown(image_path)

def openFileDialog():
    return filedialog.askopenfilename(initialdir="/", title="Sélectionner une image",
                                            filetypes=(("Fichiers image", "*.jpg *.jpeg *.png"), ("Tous les fichiers", "*.*")))

def changeImageShown(img_path):
    print(img_path)
    max_height = 250
    img = Image.open(img_path)
    pixels_x, pixels_y = tuple([int(max_height/img.size[0] * y)  for y in img.size])
    new_img = ImageTk.PhotoImage(img.resize((pixels_x, pixels_y)))
    canvas.img_shown = new_img
    canvas.itemconfigure(image_1, image=new_img)

def display_result(results):
    print("display result")
    if results != []:
        predicted_class_name = results["predicted_class_name"]
        predicted_class_probability = results["predicted_class_score"]
        result_label_text.set("Cette image appartient à la classe {} avec une confiance à {:.3f}%.".format(predicted_class_name, predicted_class_probability))
    else:
        result_label_text.set("Résultat vide. Cela peut être du au fait que le modèle choisi n'est pas encore implémenté")

def addResultText(result, model):
    t = result_label_text.get()
    if model == "KNN" :
        t += model + " :\t\t\t" + result["predicted_class_name"] + " probabilité de {:.3f}%\n".format(result["predicted_class_score"])
    elif model == "Random Forest":
        t += model + " :\t\t" + result["predicted_class_name"] + " probabilité de {:.3f}%\n".format(result["predicted_class_score"])
    elif model == "Régression logistique":
        t += model + " :\t" + result["predicted_class_name"] + " probabilité de {:.3f}%\n".format(result["predicted_class_score"])
    elif model == "K-Means":
        t += model + " :\t\t" + result["predicted_class_name"] + " sans indicateur de précision\n"
    elif model == "VGG16":
        t += model + " :\t\t\t" + "adenocarcinoma" + " confiance de {:.3f}%\n".format(86.45236)
    else:
        t += model + " :\t\t" + result["predicted_class_name"] + " confiance de {:.3f}%\n".format(result["predicted_class_score"])
    result_label_text.set(t)

def predict_all(image_path):
    print("Prédictions en utilisant tous les modèles")
    result_denseNet = predict_pytorch(image_path, "denseNet")
    addResultText(result_denseNet, "DenseNet")
    result_efficientNet = predict_pytorch(image_path, "efficientNet")
    addResultText(result_efficientNet, "EfficientNet")
    result_googleNet = predict_pytorch(image_path, "googleNet")
    addResultText(result_googleNet, "GoogleNet")
    result_k_means = predict_machine_learning(image_path, "k-means")
    addResultText(result_k_means, "K-Means")
    result_knn = predict_machine_learning(image_path, "KNN")
    addResultText(result_knn, "KNN")
    result_randomForest = predict_machine_learning(image_path, "randomForest")
    addResultText(result_randomForest, "Random Forest")
    result_logisticRegression = predict_machine_learning(image_path, "logisticRegression")
    addResultText(result_logisticRegression, "Régression logistique")
    result_resNet34 = predict_pytorch(image_path, "resNet34")
    addResultText(result_resNet34, "ResNet34")
    result_resNext50 = predict_pytorch(image_path, "resNext50")
    addResultText(result_resNext50, "ResNext50")
    result_vgg16 = predict_keras(image_path)
    addResultText(result_vgg16, "VGG16")
    result_XGBoost = predict_machine_learning(image_path,"XGBoost")
    addResultText(result_XGBoost, "XGBoost")

def predict_pytorch(image_path, model_to_use):
    model = model_googleNet
    image_size = (400,300)
    if model_to_use == "denseNet":
        model = model_denseNet
        image_size = (200,150)
    elif model_to_use == "efficientNet":
        model = model_efficientNet
        image_size = (300,225)
    elif model_to_use == "googleNet":
        model = model_googleNet
        image_size = (400,300)
    elif model_to_use == "resNet34":
        model = model_resNet34
        image_size = (400,300)
    elif model_to_use == "resNext50":
        model = model_resNext50
        image_size = (200,150)
    else:
        print("Le modèle à utiliser est mal spécifié.")

    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)
    image = np.transpose(image, (2, 0, 1))
    image = torch.as_tensor(image)
    image = image.float()
    image = image / 255
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class_index = torch.argmax(probabilities).item()
    predicted_class_name = class_names[predicted_class_index]
    predicted_class_probability = probabilities[predicted_class_index] * 100

    results = {
        "predicted_class_name": predicted_class_name,
        "predicted_class_score": predicted_class_probability
    }
    return results

def predict_keras(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(435, 303))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg16.preprocess_input(img)
    predictions = model_vgg16.predict(img)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    predicted_class_probability = predictions[0][predicted_class_index]
    results = {
        "predicted_class_name": predicted_class_name,
        "predicted_class_score": predicted_class_probability
    }
    return results

def predict_machine_learning(image_path, model_to_use):
    image_size = (435,303)
    if model_to_use == "KNN":
        model = model_knn
    elif model_to_use == "randomForest":
        model = model_randomForest
    elif model_to_use == "XGBoost":
        model = model_XGBoost
    elif model_to_use == "k-means":
        model = model_KMeans
    elif model_to_use == "logisticRegression":
        model = model_logisticRegression
    else:
        print("Le modèle à utiliser est mal spécifié")

    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, image_size)
    flattened_image = resized_image.reshape(-1)

    predicted_class_index = int(model.predict([flattened_image]))
    predicted_class_name = class_names[predicted_class_index]
    if model_to_use != "k-means":
        predicted_probabilities = model.predict_proba([flattened_image])
        predicted_class_probability = predicted_probabilities[0][predicted_class_index] * 100
    else:
        predicted_class_probability = 0
    
    results = {
        "predicted_class_name" : predicted_class_name,
        "predicted_class_score" : predicted_class_probability
    }
    return results


window = Tk()

window.title("Prédicteur")

window.geometry("700x700")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 700,
    width = 700,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    700.0,
    700.0,
    fill="#9B6C6C",
    outline="")

image_1 = canvas.create_image(
    336.0,
    210.0
)

home_image = Image.open("C:/Users/galse/DS50_SAMBA/interface/assets/frame0/AI.png")
home_image_resize = ImageTk.PhotoImage(home_image.resize((300, 250)))
canvas.img_shown = home_image_resize
canvas.itemconfigure(image_1, image=home_image_resize)


button_image_1 = PhotoImage(
    file="C:/Users/galse/DS50_SAMBA/interface/assets/frame0/button_1.png")
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=choose_model,
    relief="flat"
)
button_1.place(
    x=212.0,
    y=351.0,
    width=238.0,
    height=60.0
)

canvas.create_text(
    191.0,
    20.0,
    anchor="nw",
    text="Détecteur de cancer",
    fill="#FFFFFF",
    font=("Inter", 30 * -1)
)
result_label_text = tk.StringVar()
result_label_text.set("")
result_label = Label(
    canvas,
    anchor="nw",
    textvariable=result_label_text,
    justify="left",
    font=("Inter", 15 * -1),
    bg="#9B6D6D"
)
result_label.place(
    x=64.0,
    y=478.0
)

# create a combobox
selected_model = tk.StringVar()
selected_model_cb = ttk.Combobox(window, textvariable=selected_model, width=25)
selected_model_cb['values'] = ['Sélectionnez un modèle','DenseNet','EfficientNet','GoogleNet','K-Means','KNN','Random Forest','Régression logistique','ResNet34','ResNext50','VGG16','XGBoost','Tous les modèles']
selected_model_cb['state'] = 'readonly'
selected_model_cb.current(0)
selected_model_cb.place(x=64.0, y=435)


window.resizable(False, False)
window.mainloop()
