from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
from keras.applications import VGG16
train_dir = "C:/Users/H'z'y/Desktop/2022/program_language/python/爬虫/Fichiers Azure ML/Data_Container_Creation_Using_Storage_Explorer/train"
test_dir = "C:/Users/H'z'y/Desktop/2022/program_language/python/爬虫/Fichiers Azure ML/Data_Container_Creation_Using_Storage_Explorer/test"

conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150, 150,3))
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units = 4,activation='softmax'))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')
test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])
my_model = model.fit_generator(
    train_generator,
    steps_per_epoch=20,
    epochs=40,
    callbacks=[reduce_lr],
    validation_data=validation_generator,
    validation_steps=50)
acc = my_model.history['acc']
val_acc = my_model.history['val_acc']
loss = my_model.history['loss']
val_loss = my_model.history['val_loss']
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

import joblib
joblib.dump(model,'DT1.dat')
