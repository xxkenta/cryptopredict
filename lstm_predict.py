from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional, BatchNormalization
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import json

def test_model(window_size, X_train_shape, dropout_value, activation_function, loss_function, optimizer, weights, X_test, Y_test, unnormalized_bases):
    """
    Test the model on the testing data
    
    Arguments:
    model -- The previously fitted 3 layer Recurrent Neural Network
    X_test -- A tensor that represents the x values of the testing data
    Y_test -- A tensor that represents the y values of the testing data
    unnormalized_bases -- A tensor that can be used to get unnormalized data points
    
    Returns:
    y_predict -- A tensor that represnts the normalized values that the model predicts based on X_test
    real_y_test -- A tensor that represents the actual prices of XRP throughout the testing period
    real_y_predict -- A tensor that represents the model's predicted prices of XRP
    fig -- A branch of the graph of the real predicted prices of XRP versus the real prices of XRP
    """
    
    #Create a Sequential model using Keras
    model = Sequential()

    #First recurrent layer with dropout
    model.add(Bidirectional(LSTM(window_size, return_sequences=True), input_shape=(window_size, X_train_shape),))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))

    #Second recurrent layer with dropout
    model.add(Bidirectional(LSTM((window_size*2), return_sequences=True)))
    model.add(Dropout(dropout_value))

    #Third recurrent layer with dropout
    model.add(Bidirectional(LSTM(window_size, return_sequences=False)))
    model.add(Dropout(dropout_value))

    #Output layer (returns the predicted value)
    model.add(Dense(units=1))
    
    #Set activation function
    model.add(Activation(activation_function))

    #Set loss function and optimizer
    model.load_weights(weights)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])    
    #Test the model on X_Test
    y_predict = model.predict(X_test)

    #Create empty 2D arrays to store unnormalized values
    real_y_test = np.zeros_like(Y_test)
    real_y_predict = np.zeros_like(y_predict)

    #Fill the 2D arrays with the real value and the predicted value by reversing the normalization process
    for i in range(Y_test.shape[0]):
        y = Y_test[i]
        predict = y_predict[i]
        real_y_test[i] = (y+1)*unnormalized_bases[i]
        real_y_predict[i] = (predict+1)*unnormalized_bases[i]

    #Plot of the predicted prices versus the real prices
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_title("XRP Price Over Time")
    plt.plot(real_y_predict, color = 'green', label = 'Predicted Price')
    plt.plot(real_y_test, color = 'red', label = 'Real Price')
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Time (Hours)")
    ax.legend()
    plt.savefig("XRP-Predicted-Real-graph.png")
    
    return y_predict, real_y_test, real_y_predict, fig

weights = "models/LSTM_Final-01-0.150.model"
X_test = np.load('data/X_test.npy')
Y_test = np.load('data/Y_test.npy')
unnormalized_bases = np.load('data/unnormalized_bases.npy')

with open('stats.json') as f:
    model_values = json.load(f)
    
window_size = model_values['window_size']
X_train_shape = model_values['X_train_shape']
activation_function = model_values['activation_function']
loss_function = model_values['loss_function']
optimizer = model_values['optimizer']
dropout_value = model_values['dropout']

y_predict, real_y_test, real_y_predict, fig1 = test_model(window_size, X_train_shape, dropout_value, activation_function, loss_function, optimizer, weights, X_test, Y_test, unnormalized_bases)

plt.show(fig1)