import os
import cv2
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump
import matplotlib.pyplot as plt

train_data_path = "./images/train"
test_data_path = "./images/test"
train_classes = os.listdir(train_data_path)
test_classes = os.listdir(test_data_path)

IMAGE_HEIGHT = 435
IMAGE_WIDTH = 303

train_images = []
train_labels = []
test_images = []
test_labels = []

# chargement des images et labels d'entrainement
for i, class_name in enumerate(train_classes):
    class_path = os.path.join(train_data_path, class_name)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        image = cv2.imread(image_path)
        # Redimensionner l'image à une taille fixe (taille moyenne des images)
        resized_image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # Aplatissement de l'image en une seule dimension
        flattened_image = resized_image.reshape(-1)
        train_images.append(flattened_image)
        train_labels.append(i)

# chargement des images et labels de test
for i, class_name in enumerate(test_classes):
    class_path = os.path.join(test_data_path, class_name)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        image = cv2.imread(image_path)
        # Redimensionner l'image à une taille fixe (taille moyenne des images)
        resized_image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # Aplatissement de l'image en une seule dimension
        flattened_image = resized_image.reshape(-1)
        test_images.append(flattened_image)
        test_labels.append(i)

# Conversion des listes en tableaux NumPy
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Affectation des données d'entrainement et de test
X_train = train_images
y_train = train_labels
X_test = test_images
y_test = test_labels

# Créer un modèle Random Forest
model = RandomForestClassifier(n_estimators=400)

# Entraîner le modèle final sur les données d'entraînement
model.fit(X_train, y_train)

# Faire des prédictions sur les données de test
y_pred = model.predict(X_test)

# Évaluer les performances du modèle final
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculer la matrice de confusion
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)

# Afficher la matrice de confusion graphiquement avec les valeurs des cellules
plt.imshow(confusion, cmap='Blues')
plt.title('Matrice de Confusion')
plt.colorbar()
tick_marks = np.arange(len(test_classes))
plt.xticks(tick_marks, test_classes, rotation=45)
plt.yticks(tick_marks, test_classes)
plt.xlabel('Prédictions')
plt.ylabel('Valeurs Réelles')

# Ajouter les valeurs des cellules dans la matrice
thresh = confusion.max() / 2.0
for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        plt.text(j, i, format(confusion[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if confusion[i, j] > thresh else "black")

plt.show()

# Sauvegarder le modèle final entraîné
dump(model, 'C:/Users/galse/DS50/models/randomForest.joblib')
