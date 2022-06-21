import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator


tfk = tf.keras
tfkl = tf.keras.layers

seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset\\")

transGen = ImageDataGenerator(rotation_range=10,
                         width_shift_range = [-25, 25],
                         height_shift_range = [-25, 25],
                         zoom_range=0.1,
                         horizontal_flip=False,
                         vertical_flip=False, 
                         fill_mode='nearest')

datasetGen = ImageDataGenerator(
    validation_split = 0.2)

trainingGen = ImageDataGenerator()
validationGen = ImageDataGenerator()

trainingGen = datasetGen.flow_from_directory(directory=dataset_dir,
                                               target_size=(256,256),
                                               color_mode='rgb',
                                               classes=None,
                                               class_mode='categorical',
                                               batch_size=16,
                                               shuffle=True,
                                               seed=seed,
                                               subset = 'training')

validationGen = datasetGen.flow_from_directory(directory=dataset_dir,
                                               target_size=(256,256),
                                               color_mode='rgb',
                                               classes=None,
                                               class_mode='categorical',
                                               batch_size=16,
                                               shuffle=False,
                                               seed=seed,
                                               subset = 'validation')

input_shape = (256, 256, 3)

def build_model(input_shape):

    input = tfkl.Input(shape=input_shape, name='Input')
    x = tfkl.Rescaling(scale = 1./255, name = 'Rescaling') (input)
    x = tfkl.Conv2D(filters=64, kernel_size=(3, 3), strides = (1, 1), padding = 'same', activation = 'relu', kernel_initializer = tfk.initializers.GlorotUniform(seed), name = 'Conv64_1')(x)
    x = tfkl.Conv2D(filters=64, kernel_size=(3, 3), strides = (1, 1), padding = 'same', activation = 'relu', kernel_initializer = tfk.initializers.GlorotUniform(seed), name = 'Conv64_2')(x)
    x = tfkl.MaxPooling2D(pool_size = (2, 2), name = 'Pool_1')(x)

    x = tfkl.Conv2D(filters=32, kernel_size=(3, 3), strides = (1, 1), padding = 'same', activation = 'relu', kernel_initializer = tfk.initializers.GlorotUniform(seed), name = 'Conv32_1')(x)
    x = tfkl.Conv2D(filters=32, kernel_size=(3, 3), strides = (1, 1), padding = 'same', activation = 'relu', kernel_initializer = tfk.initializers.GlorotUniform(seed), name = 'Conv32_2')(x)
    x = tfkl.MaxPooling2D(pool_size = (2, 2), name = 'Pool_2')(x)

    x = tfkl.Conv2D(filters=16, kernel_size=(3, 3), strides = (1, 1), padding = 'same', activation = 'relu', kernel_initializer = tfk.initializers.GlorotUniform(seed), name = 'Conv16_1')(x)
    x = tfkl.Conv2D(filters=16, kernel_size=(3, 3), strides = (1, 1), padding = 'same', activation = 'relu', kernel_initializer = tfk.initializers.GlorotUniform(seed), name = 'Conv16_2')(x)
    x = tfkl.MaxPooling2D(pool_size = (2, 2), name = 'Pool_3')(x)
    
    x = tfkl.Flatten(name='Flatten')(x)
    x = tfkl.Dropout(0.3, seed=seed)(x)

    x = tfkl.Dense(units=128, activation='relu', name='Dense')(x)
    x = tfkl.Dropout(0.3, seed=seed)(x)

    output = tfkl.Dense(units=4, activation='softmax', name='Classifier')(x)

    model = tfk.Model(inputs=input, outputs=output, name='model')

    model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics='accuracy')

    return model

model = build_model(input_shape)
model.summary()

history = model.fit(
    x = trainingGen,
    batch_size = 16,
    epochs = 200,
    validation_data = validationGen,
    callbacks = [tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True)]
).history

model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model2"))