"""
models.py
---------
    Create models that will be used for classification

Current possible inputs
----------------------
Numerical data (total=43):
    Range
    Bearing
    Altitude
    Elevation
    Velocity
    Range Rate
    Signal-to-Noise Ratio
    Covariance Matrix (Maybe using a an independent model) (36 values)

Classes (one-hot, total=5)
-------
    10000
    01000
    00100
    00010
    00001

Models
-------
    1: Baseline model (dense only)
    2: 2D CNN model (dense + conv) for 2d data
    3: 2D CNN model (dense + conv) for 1d data
    3: 1D CNN model (dense + conv)
    4: Merge model (dense + conv)
"""

# TODO: add batch norm if overfitting
# TODO: scale down hidden layer dims significantly (right now they are rough drafts spit out by copilot)

# Imports
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv2D, Conv1D, MaxPooling2D, 
                                     MaxPooling1D, Input, concatenate, Activation, 
                                     BatchNormalization, LSTM)
import tensorflow.keras.optimizers as opts
import numpy as np


def dense_model(size):
    inputs = Input(shape=(size,))
    output = 5
    hl_size = (size + output) // 2

    model = Sequential()
    # input layer
    model.add(Dense(units=hl_size + 5, activation='relu', input_shape=(size,)))

    # hidden layers
    model.add(Dense(units=hl_size + 5, activation='relu'))
    model.add(Dense(units=hl_size + 5, activation='relu'))
    # add batch normalization
    # model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(units=hl_size + 5, activation='relu'))
    model.add(Dense(units=hl_size + 5, activation='relu'))
    # add batch normalization
    # model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(units=hl_size + 5, activation='relu'))
    model.add(Dense(units=hl_size + 5, activation='relu'))
    # add batch normalization
    # model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(units=hl_size + 5, activation='relu'))
    model.add(Dense(units=hl_size + 5, activation='relu'))
    # add batch normalization
    # model.add(BatchNormalization())
    model.add(Dense(units=hl_size + 5, activation='relu'))
    model.add(Dense(units=hl_size + 5, activation='relu'))
    if hl_size // 2 > 1:
        model.add(Dense(units=(hl_size // 2), activation='relu'))

    # output layer
    model.add(Dense(units=output, activation='softmax')) 

    return model
    
def GS_model(final_activ='softmax', activ='relu', hl_size=38, hl_depth=5, drop_rate=0.0, optimizer='Adam', weight_init='glorot_uniform',
             decay=.0005, momentum=0, learn_rate=.001, input_dim=52):
    input = Input(shape=(input_dim,))
    x = Dense(hl_size, activation=activ, kernel_initializer=weight_init)(input)
    for i in range(hl_depth):
        x = Dense(hl_size, activation=activ, kernel_initializer=weight_init)(x)
        x = Dropout(drop_rate)(x)
    output = Dense(5, activation=final_activ, kernel_initializer=weight_init)(x)
    model = Model(inputs=input, outputs=output)

    if optimizer == 'Adam':
        opt = opts.Adam(learning_rate=learn_rate, decay=decay)
    elif optimizer == 'SGD':
        opt = opts.SGD(learning_rate=learn_rate, momentum=momentum)
    elif optimizer == 'RMSprop':
        opt = opts.RMSprop(learning_rate=learn_rate, decay=decay)
    elif optimizer == 'Adagrad':
        opt = opts.Adagrad(learning_rate=learn_rate, decay=decay)
    elif optimizer == 'Adadelta':
        opt = opts.Adadelta(learning_rate=learn_rate, decay=decay)
    elif optimizer == 'Adamax':
        opt = opts.Adamax(learning_rate=learn_rate, decay=decay)
    elif optimizer == 'Nadam':
        opt = opts.Nadam(learning_rate=learn_rate, decay=decay)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

# 2D CNN model for 36 or 9 inputs 
# possibly add batch norm if overfitting
def cnn2D_model(width, height, kernal_size=(3, 3)):
    chanDim = -1
    inputs = Input(shape=(width, height, 1))

	# loop over the number of filters
    for (i, f) in enumerate((4, 16, 32)):
		# if this is the first CONV layer then set the input
		# appropriately
        if i == 0:
            x = inputs

		# CONV => RELU => BN => POOL
        x = Conv2D(f, kernal_size, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)

	# flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
    x = Dense(5)(x)
    x = Activation("softmax")(x)

	# construct the CNN
    model = Model(inputs, x)

    return model

# batch norm if overfitting
# use for time series
def cnn1D_model():
    model = Sequential()
    # input layer
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(52, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))

    # hidden layers
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))

    # output layer
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=5, activation='softmax'))

    return model

# possibly add batch norm if overfitting
# functional model
def merge_model():
    """
    Dense model for numerical inputs
    """
    # create model
    model1 = dense_model(52)

    """
    CNN for covariance matrix
    """
    # create model
    model2 = cnn2D_model(6, 6)

    # merge both models
    combined = concatenate([model1.output, model2.output])

    # output layer (functional) (dense)
    output_tensor = Dense(units=16, activation='relu')(combined)
    output_tensor = Dropout(0.5)(output_tensor)
    output_tensor = Dense(units=5, activation='softmax')(output_tensor)

    # create model
    model3 = Model(inputs=[model1.input, model2.input], outputs=output_tensor)

    return model3

def time_series_model(time_steps, features):
    input = Input(shape=(time_steps, features))

    hidden1 = LSTM(units=32, return_sequences=True)(input)
    hidden2 = Dropout(0.2)(hidden1)
    hidden3 = LSTM(units=32, return_sequences=False)(hidden2)
    hidden4 = Dropout(0.2)(hidden3)
    
    output = Dense(units=5, activation='softmax')(hidden4)

    model = Model(inputs=input, outputs=output)

    return model