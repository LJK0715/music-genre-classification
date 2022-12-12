import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import os
import librosa
import librosa.display
from IPython.display import Audio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import keras as k
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('features_3_sec.csv')
df = df.drop(labels = 'filename', axis = 1)

class_list = df.iloc[:, -1]
convertor = LabelEncoder()
y = convertor.fit_transform(class_list)
fit = StandardScaler()
X = fit.fit_transform(np.array(df.iloc[:, :-1],dtype = float))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

def trainModel(model, epochs, optimizer):
    batch_size = 128
    #callback = myCallback()
    model.compile(optimizer = optimizer,
                  loss = 'sparse_categorical_crossentropy',
                  metrics = 'accuracy')
    return model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = epochs,
                     batch_size = batch_size)

def plotValidate(history):
    print('validation accuracy', max(history.history['val accuracy']))
    pd.DataFrame(history.history).plot(figsize = (12, 6))
    plt.show()

model = k.models.Sequential([
    k.layers.Dense(512, activation = 'relu', input_shape = (X_train.shape[1],)),
    k.layers.Dropout(0.2),
    
    k.layers.Dense(256, activation = 'relu'),
    k.layers.Dropout(0.2),
    
    k.layers.Dense(128, activation = 'relu'),
    k.layers.Dropout(0.2),
    
    k.layers.Dense(64, activation = 'relu'),
    k.layers.Dropout(0.2),
    
    k.layers.Dense(10, activation = 'softmax'),
])

print(model.summary())
model_history = trainModel(model = model, epochs = 600, optimizer = 'adam')

test_loss, test_acc = model.evaluate(X_test, y_test, batch_size = 128)
print("The test Loss is:", test_loss)
print("\nThe Best test Accuracy is :", test_acc*100)