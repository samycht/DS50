from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
import os
from keras import optimizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# data preprocessing
# When training, you need to change this to your own data path
train_dir = "C:/Users/H'z'y/Desktop/2022/program_language/python/爬虫/Fichiers Azure ML/Data_Container_Creation_Using_Storage_Explorer/train"
val_dir = "C:/Users/H'z'y/Desktop/2022/program_language/python/爬虫/Fichiers Azure ML/Data_Container_Creation_Using_Storage_Explorer/test"

# use data augmentation
#train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40., width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
#val_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40., width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Do not use data augmentation
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
# Generate image tensors using iterators
train_gen = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=30)
val_gen = train_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=30)

model = models.Sequential()
# uses 3x3 sliding convolution
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(224, 224, 3)))
# 2 x 2 MaxPool
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# expand to a one-dimensional vector
model.add(layers.Flatten())
# Use dropout operation to reduce overfitting
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))
model.summary()
# compile model
model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['acc'])
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
# training model
history = model.fit_generator(train_gen, epochs=25, validation_data=val_gen,callbacks=[reduce_lr])



# Plot training accuracy loss curve
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='training acc')
plt.plot(epochs, val_acc, 'r', label='val acc')
plt.title('training & val accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='training loss')
plt.plot(epochs, val_loss, 'r', label='val loss')
plt.title('training & val loss')
plt.legend()
plt.show()