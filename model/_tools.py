# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import numpy as np

import keras
from keras.models import Model
from keras.models import Sequential

from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
from keras.layers import Conv3D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import MaxPooling3D
from keras.layers import MaxPooling2D

from deepsolar.layers import JANet
from deepsolar.layers import ConvJANet

from keras.optimizers import Adam

from keras.utils import np_utils
from keras.utils import plot_model

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

# -----------------------------------------------------------------------------
# crear arquitectura LSTM-Dense
def build_lstm_model(n_input, n_output, n_feature, lstm_type,
                     lstm_layers_1, lstm_units_1, lstm_activation_1,
                     dense_layers_1, dense_units_1, dense_activation_1,
                     dense_layers_2, dense_units_2, dense_activation_2,
                     dense_layers_3, dense_units_3, dense_activation_3,
                     dropout_rate=0.2):
    
    # inicializamos sys_model -------------------------------------------------

    # input layer
    input_sys = Input( shape=(n_input, n_feature) )
    
    # LSTM layers
    sys_model = lstm_type(units=lstm_units_1, activation=lstm_activation_1, return_sequences=True)(input_sys)
    
    for i in np.arange(lstm_layers_1 - 1):
        sys_model = lstm_type(units=lstm_units_1, activation=lstm_activation_1, return_sequences=True)(sys_model)
    #sys_model = Dropout(rate=dropout_rate)(sys_model)
        
    # Dense_1 layers
    for i in np.arange(dense_layers_1):
        sys_model = Dense(units=dense_units_1, activation=dense_activation_1)(sys_model)
    sys_model = Dropout(rate=dropout_rate)(sys_model)
    
    # Dense_2 layers
    for i in np.arange(dense_layers_2):
        sys_model = Dense(units=dense_units_2, activation=dense_activation_2)(sys_model)

    # Flatten
    sys_model = Flatten()(sys_model)
    sys_model = Dropout(rate=dropout_rate)(sys_model)
    
    # Dense_3 layers
    for i in np.arange(dense_layers_3):
        sys_model = Dense(units=dense_units_3, activation=dense_activation_3)(sys_model)

    # ouput
    output_sys = Dense(units=n_output, activation='linear')(sys_model)
    
    # retornar modelo
    forecasting_model = Model(inputs=input_sys, outputs=output_sys)
    
    return forecasting_model

def build_lstm_conv_model(n_input, n_output, n_feature, img_size,
                          lstm_type, conv_type,
                          lstm_layers_1, lstm_units_1, lstm_activation_1,
                          dense_layers_1, dense_units_1, dense_activation_1,
                          dense_layers_2, dense_units_2, dense_activation_2,
                          dense_layers_3, dense_units_3, dense_activation_3,
                          conv_layers_1, conv_filters_1, conv_activation_1,
                          conv_layers_2, conv_filters_2, conv_activation_2,
                          conv_layers_3, conv_filters_3, conv_activation_3,
                          dense_layers_4, dense_units_4, dense_activation_4,
                          dense_layers_5, dense_units_5, dense_activation_5, 
                          dropout_rate=0.2,
                          **kargs):
    
    # inicializamos sys_model -------------------------------------------------

    # input layer
    input_sys = Input( shape=(n_input, n_feature) )
    
    # LSTM layers
    sys_model = lstm_type(units=lstm_units_1, activation=lstm_activation_1, return_sequences=True)(input_sys)
    
    for i in np.arange(lstm_layers_1 - 1):
        sys_model = lstm_type(units=lstm_units_1, activation=lstm_activation_1, return_sequences=True)(sys_model)
    #sys_model = Dropout(rate=dropout_rate)(sys_model)
        
    # Dense_1 layers
    for i in np.arange(dense_layers_1):
        sys_model = Dense(units=dense_units_1, activation=dense_activation_1)(sys_model)
    sys_model = Dropout(rate=dropout_rate)(sys_model)
    
    # Dense_2 layers
    for i in np.arange(dense_layers_2):
        sys_model = Dense(units=dense_units_2, activation=dense_activation_2)(sys_model)

    # Flatten
    sys_model = Flatten()(sys_model)
    sys_model = Dropout(rate=dropout_rate)(sys_model)
    
    # Dense_3 layers
    for i in np.arange(dense_layers_3 - 1):
        sys_model = Dense(units=dense_units_3, activation=dense_activation_3)(sys_model)

    output_sys = Dense(units=dense_units_3, activation=dense_activation_3)(sys_model)
    
    # inicializamos g16_model -------------------------------------------------
    
    # Input
    input_g16 = Input( shape=(n_input, img_size, img_size, 1) )
    
    # ConvLSTM_1
    g16_model = conv_type(filters=conv_filters_1, kernel_size=(3, 3),
                             activation=conv_activation_1,
                             padding='same', return_sequences=True)(input_g16)
    for i in np.arange(conv_layers_1 - 1):
        g16_model = conv_type(filters=conv_filters_1, kernel_size=(5, 5),
                              activation=conv_activation_1,
                              padding='same', return_sequences=True)(g16_model)
    g16_model = MaxPooling3D(pool_size=(1,2,2))(g16_model)
    #g16_model = BatchNormalization()(g16_model)
    
    # ConvLSTM_2
    for i in np.arange(conv_layers_2):
        g16_model = conv_type(filters=conv_filters_2, kernel_size=(3, 3),
                              activation=conv_activation_2,
                              padding='same', return_sequences=True)(g16_model)
    g16_model = MaxPooling3D(pool_size=(1,2,2))(g16_model)
    #g16_model = BatchNormalization()(g16_model)
    
    # ConvLSTM_3
    for i in np.arange(conv_layers_3):
        g16_model = conv_type(filters=conv_filters_3, kernel_size=(3, 3),
                              activation=conv_activation_3,
                              padding='same', return_sequences=True)(g16_model)
    #g16_model = BatchNormalization()(g16_model)
    
    # Flatten
    g16_model = Conv3D(filters=1, kernel_size=(3, 3, 3),
                       activation='relu', padding='same', data_format='channels_last')
    g16_model = Flatten()(g16_model)
    g16_model = Dropout(rate=dropout_rate)(g16_model)
    
    # Dense_4 layers
    for i in np.arange(dense_layers_4 - 1):
        g16_model = Dense(units=dense_units_4, activation=dense_activation_4)(g16_model)
    output_g16 = Dense(units=dense_units_4, activation=dense_activation_4)(g16_model)
    
    # concatenamos los modelos para el modelo final
    concat_layer = Concatenate()([output_sys, output_g16])
    
    # Dense_5 layers
    for i in np.arange(dense_layers_5):
        concat_layer = Dense(units=dense_units_5, activation=dense_activation_5)(concat_layer)
    
    model_output = Dense(units=n_output, activation = 'linear')(concat_layer)
    
    forecasting_model = Model(inputs=[input_sys, input_g16], outputs=model_output)
    
    # retornar modelo
    return forecasting_model

def build_lstm_conv2d_model(n_input, n_output, n_feature, img_size,
                          lstm_type,
                          lstm_layers_1, lstm_units_1, lstm_activation_1,
                          dense_layers_1, dense_units_1, dense_activation_1,
                          dense_layers_2, dense_units_2, dense_activation_2,
                          dense_layers_3, dense_units_3, dense_activation_3,
                          conv_layers_1, conv_filters_1, conv_activation_1,
                          conv_layers_2, conv_filters_2, conv_activation_2,
                          conv_layers_3, conv_filters_3, conv_activation_3,
                          dense_layers_4, dense_units_4, dense_activation_4,
                          dense_layers_5, dense_units_5, dense_activation_5, 
                          dropout_rate=0.2,
                          **kargs):
    
    # inicializamos sys_model -------------------------------------------------

    # input layer
    input_sys = Input( shape=(n_input, n_feature) )
    
    # LSTM layers
    sys_model = lstm_type(units=lstm_units_1, activation=lstm_activation_1, return_sequences=True)(input_sys)
    
    for i in np.arange(lstm_layers_1 - 1):
        sys_model = lstm_type(units=lstm_units_1, activation=lstm_activation_1, return_sequences=True)(sys_model)
    #sys_model = Dropout(rate=dropout_rate)(sys_model)
        
    # Dense_1 layers
    for i in np.arange(dense_layers_1):
        sys_model = Dense(units=dense_units_1, activation=dense_activation_1)(sys_model)
    sys_model = Dropout(rate=dropout_rate)(sys_model)
    
    # Dense_2 layers
    for i in np.arange(dense_layers_2):
        sys_model = Dense(units=dense_units_2, activation=dense_activation_2)(sys_model)

    # Flatten
    sys_model = Flatten()(sys_model)
    sys_model = Dropout(rate=dropout_rate)(sys_model)
    
    # Dense_3 layers
    for i in np.arange(dense_layers_3 - 1):
        sys_model = Dense(units=dense_units_3, activation=dense_activation_3)(sys_model)

    output_sys = Dense(units=dense_units_3, activation=dense_activation_3)(sys_model)
    
    # inicializamos g16_model -------------------------------------------------
    
    # Input
    input_g16 = Input( shape=(img_size, img_size, 1) )
    
    # ConvLSTM_1
    g16_model = Conv2D(filters=conv_filters_1, kernel_size=(3, 3),
                       activation=conv_activation_1,
                       padding='same', return_sequences=True)(input_g16)
    for i in np.arange(conv_layers_1 - 1):
        g16_model = Conv2D(filters=conv_filters_1, kernel_size=(5, 5),
                           activation=conv_activation_1,
                           padding='same', return_sequences=True)(g16_model)
    g16_model = MaxPooling2D(pool_size=(2,2))(g16_model)
    #g16_model = BatchNormalization()(g16_model)
    
    # ConvLSTM_2
    for i in np.arange(conv_layers_2):
        g16_model = Conv2D(filters=conv_filters_2, kernel_size=(3, 3),
                           activation=conv_activation_2,
                           padding='same', return_sequences=True)(g16_model)
    g16_model = MaxPooling2D(pool_size=(2,2))(g16_model)
    #g16_model = BatchNormalization()(g16_model)
    
    # ConvLSTM_3
    for i in np.arange(conv_layers_3):
        g16_model = Conv2D(filters=conv_filters_3, kernel_size=(3, 3),
                           activation=conv_activation_3,
                           padding='same', return_sequences=True)(g16_model)
    #g16_model = BatchNormalization()(g16_model)
    
    # Flatten
    g16_model = Flatten()(g16_model)
    g16_model = Dropout(rate=dropout_rate)(g16_model)
    
    # Dense_4 layers
    for i in np.arange(dense_layers_4 - 1):
        g16_model = Dense(units=dense_units_4, activation=dense_activation_4)(g16_model)
    output_g16 = Dense(units=dense_units_4, activation=dense_activation_4)(g16_model)
    
    # concatenamos los modelos para el modelo final
    concat_layer = Concatenate()([output_sys, output_g16])
    
    # Dense_5 layers
    for i in np.arange(dense_layers_5):
        concat_layer = Dense(units=dense_units_5, activation=dense_activation_5)(concat_layer)
    
    model_output = Dense(units=n_output, activation = 'linear')(concat_layer)
    
    forecasting_model = Model(inputs=[input_sys, input_g16], outputs=model_output)
    
    # retornar modelo
    return forecasting_model

# -----------------------------------------------------------------------------
# optimizar modelo medainte grid search
def optimize_model_structure(build_fn, param_grid, X_train, X_test, Y_train, Y_test):
    
    # setear método de inicialización
    model = KerasRegressor(build_fn=build_fn)
    
    # inicializar grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X_train, Y_train)
    
    print("Best score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
    # retornar mejor modelo
    return grid_result.best_estimator_ 