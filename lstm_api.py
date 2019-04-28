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
import requests
import json
import inspect


def load_data(sequence_length, percent_training, crypto, exchange):
    """
    Loads the XRP data
    
    Arguments:
    sequence_length -- An integer of how many days should be looked at in a row
    percent_training -- a float from 0.1-0.9 to how much data should be used as training data
    crypto -- a string for the crypto name
    exchange -- a string for the exchange name
    
    
    Returns:
    X_train -- A tensor that will be inputed into the model to train it
    Y_train -- A tensor that will be inputed into the model to train it
    X_test -- A tensor that will be used to test the model's proficiency
    Y_test -- A tensor that will be used to check the model's predictions
    Y_daybefore -- A tensor that represents the price of XRP the day before each Y_test value
    unnormalized_bases -- A tensor that will be used to get the true prices from the normalized ones
    window_size -- An integer that represents how many days of X values the model can look at at once
    timerange -- A list containing the starting time and end time of the data
    """

    api_key = "ad8562fd1f3270335973682f6705bc89d621e3c0b12bb8de72c602b2c4a31300"
    limit="2000"
    
    response = requests.get("https://min-api.cryptocompare.com/data/histohour?fsym=" + crypto + "&tsym=USD&limit=" + limit + "&e=" + exchange + "&api_key=" + api_key,verify=False)
    hist = pd.DataFrame(json.loads(response.content)['Data'])
    for i in range(4):
        timestamp = str(hist['time'].iloc[0])
        response = requests.get("https://min-api.cryptocompare.com/data/histohour?fsym=" + crypto + "&tsym=USD&limit=" + limit + "&toTs=" + timestamp + "&e=" + exchange + "&api_key=" + api_key,verify=False)
        hist = hist.append(pd.DataFrame(json.loads(response.content)['Data']),ignore_index=True)
        hist = hist.sort_values(by=['time'])
    
    hist = hist.reset_index(drop=True)
    #get the first and last time point from the series
    timerange= hist['time']
    timerange = pd.to_datetime(timerange, unit='s')
    timerange = timerange.iloc[[0, -1]]
    timerange = timerange.values
    
    hist = hist.values

    #Change all zeros to the number before the zero occurs

    for x in range(0, hist.shape[0]):
        for y in range(0, hist.shape[1]):
            if(hist[x][y] == 0):
                hist[x][y] = hist[x-1][y]
    
    #Convert the file to a list
    data = hist.tolist()
        
    #Convert the data to a 3D array (a x b x c) 
    #Where a is the number of days, b is the window size, and c is the number of features in the data file
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    #Normalizing data by going through each window
    #Every value in the window is divided by the first value in the window, and then 1 is subtracted
    d0 = np.array(result)
    dr = np.zeros_like(d0)
    dr[:,1:,:] = d0[:,1:,:] / d0[:,0:1,:] - 1
    
    #Keeping the unnormalized prices for Y_test
    #Useful when graphing price over time later
    #start = beginning of validation data (for example, if dataset is 20,000, start will be 18,000).. 0.9 means get last 10% of training set
    start = int(len(result)*percent_training)
    #end = end of validation data
    #d0[start:end,0:1,column_of_close_price]
    end = int(dr.shape[0] + 1)
    unnormalized_bases = d0[start:end,0:1,1]
    
    #Splitting data set into training and testing data
    split_line = round(percent_training * dr.shape[0])
    training_data = dr[:int(split_line), :]
    
    #Shuffle the data
    np.random.shuffle(training_data)
    
    #Training Data
    X_train = training_data[:, :-1]
    Y_train = training_data[:, -1]
    Y_train = Y_train[:, 2]
    
    #Testing data
    X_test = dr[int(split_line):, :-1]
    Y_test = dr[int(split_line):, 49, :]
    Y_test = Y_test[:, 2]

    #Get the day before Y_test's price
    Y_daybefore = dr[int(split_line):, 48, :]
    Y_daybefore = Y_daybefore[:, 6]
    
    #Get window size and sequence length
    window_size = sequence_length - 1 #because the last value is reserved as the y value
    
    return X_train, Y_train, X_test, Y_test, Y_daybefore, unnormalized_bases, window_size, timerange

def train_model(X_train, Y_train, batch_num, num_epoch, X_test, Y_test, window_size, dropout_value, activation_function, loss_function, optimizer):
    """
    Initializes and creates the model to be used
    
    Arguments:
    window_size -- An integer that represents how many days of X_values the model can look at at once
    dropout_value -- A decimal representing how much dropout should be incorporated at each level, in this case 0.2
    activation_function -- A string to define the activation_function, in this case it is linear
    loss_function -- A string to define the loss function to be used, in the case it is mean squared error
    optimizer -- A string to define the optimizer to be used, in the case it is adam
    
    Returns:
    model -- A 3 layer RNN with 100*dropout_value dropout in each layer that uses activation_function as its activation
             function, loss_function as its loss function, and optimizer as its optimizer
    """
    #Create a Sequential model using Keras
    model = Sequential()

    #First recurrent layer with dropout
    model.add(Bidirectional(LSTM(window_size, return_sequences=True), input_shape=(window_size, X_train.shape[-1]),))
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
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    filepath = "LSTM_Final-{epoch:02d}-{val_acc:.3f}"
    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode="max")) #saves only the best model from each epoch   
    
    #Record the time the model starts training
    start = time.time()

    #Train the model on X_train and Y_train
    model.fit(X_train, Y_train, batch_size= batch_num, nb_epoch=num_epoch, validation_data=(X_test, Y_test), callbacks=[checkpoint])

    #Get the time it took to train the model (in seconds)
    training_time = int(math.floor(time.time() - start))
    processed_per_second = float(len(X_train)/training_time)    
    
    return model, training_time, processed_per_second

def fit_model(model, X_train, Y_train, batch_num, num_epoch, X_test, Y_test):
    """
    Fits the model to the training data
    
    Arguments:
    model -- The previously initalized 3 layer Recurrent Neural Network
    X_train -- A tensor that represents the x values of the training data
    Y_train -- A tensor that represents the y values of the training data
    batch_num -- An integer representing the batch size to be used
    num_epoch -- An integer defining the number of epochs to be run
    X_test, Y_test -- Tensors used to calculate validation accuracy after each epoch
    
    Returns:
    model -- The 3 layer Recurrent Neural Network that has been fitted to the training data
    training_time -- An integer representing the amount of time (in seconds) that the model was training
    process_per_second -- a float representing the amount of data points processed per second
    """
    filepath = "LSTM_Final-{epoch:02d}-{val_acc:.3f}"
    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode="max")) #saves only the best model from each epoch   
    #Record the time the model starts training
    start = time.time()

    #Train the model on X_train and Y_train
    model.fit(X_train, Y_train, batch_size= batch_num, nb_epoch=num_epoch, validation_data=(X_test, Y_test), callbacks=[checkpoint])

    #Get the time it took to train the model (in seconds)
    training_time = int(math.floor(time.time() - start))
    process_per_second = float(len(X_train)/training_time)
    
    print (model.summary())
    return model, training_time, process_per_second

def json_export(window_size, X_train_shape, activation_function, loss_function, optimizer, sequence_length, dropout, epochs, percent_training, batch_size, training_time, processed_per_second):
    stats = {}
    stats['window_size'] = window_size
    stats['X_train_shape'] = X_train_shape
    stats['activation_function'] = activation_function
    stats['loss_function'] = loss_function
    stats['optimizer'] = optimizer
    stats['sequence_length'] = sequence_length
    stats['dropout'] = dropout
    stats['epochs'] = epochs
    stats['percent_training'] = percent_training
    stats['batch_size'] = batch_size
    stats['training_time'] = training_time
    stats['processed_per_second'] = processed_per_second
    
    with open('stats.json', 'w') as outfile:  
        json.dump(stats, outfile)    
    return
    
def test_model(model, X_test, Y_test, unnormalized_bases):
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

crypto = "XRP"
exchange = "Bittrex"

sequence_length = 50
dropout = 0.2
epochs = 1
percent_training = 0.9
batch_size=1024
activation_function = 'relu'
loss_function = 'mse'
optimizer = 'adam'

X_train, Y_train, X_test, Y_test, Y_daybefore, unnormalized_bases, window_size, timerange = load_data(sequence_length, percent_training, crypto, exchange)

#Initializing the Model
model, training_time, processed_per_second = train_model(
    X_train, Y_train, 
    batch_size, epochs, 
    X_test, Y_test, 
    window_size, dropout, activation_function, loss_function, optimizer)

#Print the training time
print ("Training time", training_time, "seconds")
print ("Data points processed per second", processed_per_second)

json_export(window_size,
            X_train.shape[-1], 
            activation_function, 
            loss_function, 
            optimizer, 
            sequence_length, 
            dropout, 
            epochs, 
            percent_training, 
            batch_size, 
            training_time, 
            processed_per_second)

np.save('data/X_test', X_test)
np.save('data/Y_test', Y_test)
np.save('data/unnormalized_bases', unnormalized_bases)

#Testing the Model
#y_predict, real_y_test, real_y_predict, fig1 = test_model(model, X_test, Y_test, unnormalized_bases)

#Show the plot
#plt.show(fig1)

