# Import the necessary packages
import os
import glob
import random
import shutil
import warnings
import itertools
import numpy as np
import configparser
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow import keras
from plotImages import plotImages
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from confusion_matrix_plot import plot_confusion_matrix
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D

warnings.simplefilter(action='ignore', category=FutureWarning)

# Get values from a config file and update variables in the config directory
configParser = configparser.RawConfigParser()
configFilePath = r'/home/anirudh/Documents/Keras_tutorials/param.conf'
configParser.read(configFilePath)

# Declaring variables for the param.conf file
train_flag = configParser.getboolean('flags', 'train')
epochs = configParser.getint('hyper-parameters', 'epochs')
learning_rate = configParser.getfloat('hyper-parameters', 'learning_rate')
verbose = configParser.getint('hyper-parameters', 'verbose')
batch_size = configParser.getint('hyper-parameters', 'batch_size')
train_steps = configParser.getint('hyper-parameters', 'train_steps')
valid_steps = configParser.getint('hyper-parameters', 'valid_steps')

# Organise data into trin test ad valid data sets
os.chdir('/home/anirudh/Documents/Keras_tutorials/dogs-vs-cats')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')

    for c in random.sample(glob.glob('train/cat*'), 500):
        shutil.move(c, 'train/cat')
    for c in random.sample(glob.glob('train/dog*'), 500):
        shutil.move(c, 'train/dog')
    for c in random.sample(glob.glob('train/cat*'), 100):
        shutil.move(c, 'valid/cat')
    for c in random.sample(glob.glob('train/dog*'), 100):
        shutil.move(c, 'valid/dog')
    for c in random.sample(glob.glob('train/cat*'), 50):
        shutil.move(c, 'test/cat')
    for c in random.sample(glob.glob('train/dog*'), 50):
        shutil.move(c, 'test/dog')
os.chdir('../../')

# Preprocessing the data using the VGG16 pre processing
train_path = '/home/anirudh/Documents/Keras_tutorials/dogs-vs-cats/train'
valid_path = '/home/anirudh/Documents/Keras_tutorials/dogs-vs-cats/valid'
test_path = '/home/anirudh/Documents/Keras_tutorials/dogs-vs-cats/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10, shuffle=False)

assert train_batches.n == 1000
assert valid_batches.n == 200
assert test_batches.n == 100
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2

# Print images
imgs, labels = next(train_batches)
plotImages(imgs, 'first_img')
print(imgs.shape)
print(labels)

imgs, labels = next(valid_batches)
plotImages(imgs, 'second_img')
print(imgs.shape)
print(labels)

# Initialise a sequential model
vgg16_model = tf.keras.applications.vgg16.VGG16()

vgg16_model.summary()

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

model.summary()
for layer in model.layers:
    layer.trainable = False

model.add(Dense(units=2, activation='softmax'))
model.summary()

# Train the model
if train_flag is True:
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    model.fit(x=train_batches, validation_data=valid_batches,
              steps_per_epoch=train_steps, validation_steps=valid_steps,
              epochs=epochs, shuffle=True, verbose=verbose)

# Save the Model
if os.path.isfile('/home/anirudh/Documents/Keras_tutorials/dogs-vs-cats/vgg16.h5') is False:
    model.save('/home/anirudh/Documents/Keras_tutorials/dogs-vs-cats/vgg16.h5')

# Prediction usng the daved model
print("testing in progress")
print(test_batches.classes)
new_model = load_model('/home/anirudh/Documents/Keras_tutorials/dogs-vs-cats/vgg16.h5')
print(new_model.summary())
prediction = new_model.predict(x=test_batches,verbose=1, steps=10)
print(np.round(prediction))
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(prediction,axis=-1))
cm_plot_labels = ['cat','dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
